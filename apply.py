import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 나의 주식 포트폴리오 대시보드 (원/달러 통합형)")

# ==========================================
# 1. 모든 함수는 최상단에 독립적으로 선언합니다.
# ==========================================

def is_korean_stock(ticker):
    t = str(ticker).strip().upper()
    return t.endswith(('.KS', '.KQ', '.KR')) or t.replace('.KS', '').replace('.KQ', '').isdigit()

@st.cache_data(ttl=86400)
def get_company_names(tickers):
    mapping = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            name = info.get('shortName', info.get('longName', ticker))
            mapping[ticker] = name
        except Exception:
            mapping[ticker] = ticker
    return mapping

# [수정됨] 조건문 안에 있던 주가 수집 함수를 바깥으로 빼냈습니다.
@st.cache_data(ttl=3600)
def load_market_data(tickers, start):
    query_tickers = [t + '.KS' if str(t).isdigit() else t for t in tickers]
    stock_data = yf.download(query_tickers, start=start, progress=False)['Close']
    if isinstance(stock_data, pd.Series):
        stock_data = stock_data.to_frame(name=tickers[0])
    else:
        # 섞인 컬럼 이름을 원래 티커로 1:1 매칭 복구
        rename_dict = {q: t for q, t in zip(query_tickers, tickers)}
        stock_data = stock_data.rename(columns=rename_dict)
        
    fx_data = yf.download('KRW=X', start=start, progress=False)['Close']
    if isinstance(fx_data, pd.DataFrame):
        fx_data = fx_data.squeeze()
        
    if stock_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(None)
    if fx_data.index.tz is not None:
        fx_data.index = fx_data.index.tz_localize(None)
        
    return stock_data, fx_data

@st.cache_data(ttl=86400)
def get_split_events(tickers, start_date):
    split_records = []
    for ticker in tickers:
        try:
            query_ticker = ticker + '.KS' if str(ticker).isdigit() else ticker
            splits = yf.Ticker(query_ticker).splits
            
            if not splits.empty:
                splits.index = splits.index.tz_localize(None)
                start_pd = pd.to_datetime(start_date).tz_localize(None)
                splits = splits[splits.index >= start_pd]
                
                for date, ratio in splits.items():
                    if ratio > 0 and ratio != 1.0:
                        split_records.append({
                            'Date': date.normalize(),
                            'Ticker': ticker, 
                            'Ratio': ratio    
                        })
        except Exception:
            continue
    return pd.DataFrame(split_records)

def get_portfolio_snapshot(df_trades, target_date, fx_series, name_mapping, market_data):
    target_date_pd = pd.to_datetime(target_date)
    
    df_combined = df_trades[df_trades['Date'] <= target_date_pd].copy()
    df_combined['SortOrder'] = df_combined['Type'].map({'Split': 1, 'Buy': 2, 'Sell': 3})
    df = df_combined.sort_values(['Date', 'SortOrder']).copy()
    
    portfolio = {} 
    
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        t_type = row['Type']
        qty = row['Qty']
        price = row['Price']
        date = row['Date']
        
        fx_rate = fx_series.asof(date) if not pd.isna(fx_series.asof(date)) else 1300.0
        exchange_multiplier = 1 if is_korean_stock(ticker) else fx_rate
        price_krw = price * exchange_multiplier
        
        if ticker not in portfolio:
            portfolio[ticker] = {'qty': 0.0, 'total_cost_krw': 0.0, 'avg_cost_krw': 0.0}
            
        pos = portfolio[ticker]
        
        if t_type == 'Split':
            if pos['qty'] > 0:
                pos['qty'] *= qty           
                pos['avg_cost_krw'] /= qty  

        elif t_type == 'Buy':
            pos['total_cost_krw'] += qty * price_krw
            pos['qty'] += qty
            if pos['qty'] > 0:
                pos['avg_cost_krw'] = pos['total_cost_krw'] / pos['qty']
                
        elif t_type == 'Sell':
            if pos['qty'] > 0:
                pos['total_cost_krw'] -= qty * pos['avg_cost_krw']
                pos['qty'] -= qty
                if pos['qty'] <= 1e-4: 
                    pos['qty'] = 0.0
                    pos['avg_cost_krw'] = 0.0
                    pos['total_cost_krw'] = 0.0
                    
    target_prices = market_data.loc[target_date_pd] if target_date_pd in market_data.index else market_data.asof(target_date_pd)
    target_fx = fx_series.asof(target_date_pd) if not pd.isna(fx_series.asof(target_date_pd)) else 1300.0

    current_holdings = []
    for ticker, pos in portfolio.items():
        if pos['qty'] > 1e-4:
            price_on_target_date = target_prices.get(ticker, 0.0)
            exchange_multiplier = 1 if is_korean_stock(ticker) else target_fx 
            
            current_val = pos['qty'] * price_on_target_date * exchange_multiplier
            invested_val = pos['total_cost_krw']
            profit = current_val - invested_val
            return_rate = (profit / invested_val * 100) if invested_val > 0 else 0.0
            
            current_holdings.append({
                '종목코드': ticker,
                '종목명': name_mapping.get(ticker, ticker),
                '매수 단가(KRW)': pos['avg_cost_krw'],
                '보유 수량': pos['qty'],
                '보유 금액(KRW)': current_val,
                '수익금(KRW)': profit,
                '수익률 (%)': return_rate
            })
            
    return pd.DataFrame(current_holdings)

def calculate_realized_profit(df_trades, fx_series, name_mapping):
    df_combined = df_trades.copy()
    df_combined['SortOrder'] = df_combined['Type'].map({'Split': 1, 'Buy': 2, 'Sell': 3})
    df = df_combined.sort_values(['Date', 'SortOrder']).copy()
    
    portfolio = {}
    realized_records = [] 
    
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        t_type = row['Type']
        qty = row['Qty']
        price = row['Price']
        date = row['Date']
        
        fx_rate = fx_series.asof(date) if not pd.isna(fx_series.asof(date)) else 1300.0
        exchange_multiplier = 1 if is_korean_stock(ticker) else fx_rate
        price_krw = price * exchange_multiplier
        
        if ticker not in portfolio:
            portfolio[ticker] = {'qty': 0.0, 'total_cost_krw': 0.0, 'avg_cost_krw': 0.0}
            
        pos = portfolio[ticker]

        if t_type == 'Split':
            if pos['qty'] > 0:
                pos['qty'] *= qty           
                pos['avg_cost_krw'] /= qty  

        elif t_type == 'Buy':
            total_cost_krw = (pos['qty'] * pos['avg_cost_krw']) + (qty * price_krw)
            pos['qty'] += qty
            if pos['qty'] > 0:
                pos['avg_cost_krw'] = total_cost_krw / pos['qty']
                
        elif t_type == 'Sell':
            if pos['qty'] > 0:
                profit_krw = (price_krw - pos['avg_cost_krw']) * qty
                profit_rate = ((price_krw - pos['avg_cost_krw']) / pos['avg_cost_krw']) * 100
                
                realized_records.append({
                    '매도 일자': date.date(),
                    '종목명': name_mapping.get(ticker, ticker),
                    '매도 단가(KRW)': price_krw,
                    '매수 평단가(KRW)': pos['avg_cost_krw'],
                    '매도 수량': qty,
                    '실현 수익금(KRW)': profit_krw,
                    '수익률 (%)': profit_rate
                })
                pos['qty'] -= qty
                
    return pd.DataFrame(realized_records)

# ==========================================
# 2. 메인 어플리케이션 실행부
# ==========================================
st.sidebar.markdown("**[작성 규칙]**\n- 한국 주식: 원화(KRW) 단가 입력\n- 미국 주식: 달러(USD) 단가 입력")
uploaded_file = st.sidebar.file_uploader("매매 이력 CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    df_trade = pd.read_csv(uploaded_file)
    
    df_trade['Date'] = pd.to_datetime(df_trade['Date'])
    df_trade['Qty'] = pd.to_numeric(df_trade['Qty'].astype(str).str.replace(',', '').str.strip())
    df_trade['Price'] = pd.to_numeric(df_trade['Price'].astype(str).str.replace(',', '').str.strip())
    df_trade['Type'] = df_trade['Type'].astype(str).str.strip().str.capitalize()
    
    tickers = df_trade['Ticker'].unique().tolist()
    start_date = df_trade['Date'].min()
    end_date = pd.Timestamp.today().normalize()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    st.sidebar.success("데이터 로드 완료!")
        
    with st.spinner("종목명, 주가 및 환율 데이터를 불러오는 중입니다..."):
        name_mapping = get_company_names(tickers)
        # [안전장치] 캐시에서 꺼내올 때 .copy()를 사용하여 원본 오염을 완벽히 차단합니다.
        market_data_raw, fx_data_raw = load_market_data(tickers, start_date)
        market_data = market_data_raw.copy()
        fx_data = fx_data_raw.copy()
        
        df_splits = get_split_events(tickers, start_date) 
        
    market_data = market_data.reindex(all_dates).ffill().bfill()
    fx_data = fx_data.reindex(all_dates).ffill().bfill()
    
    if not df_splits.empty:
        for _, row in df_splits.iterrows():
            split_date = row['Date']
            ticker = row['Ticker']
            ratio = row['Ratio']
            
            if ticker in market_data.columns:
                mask = market_data.index < split_date
                market_data.loc[mask, ticker] *= ratio

    df_all_events = df_trade.copy()
    df_all_events['SortOrder'] = df_all_events['Type'].map({'Split': 1, 'Buy': 2, 'Sell': 3})
    df_all_events = df_all_events.sort_values(['Date', 'SortOrder'])

    daily_qty = pd.DataFrame(index=all_dates, columns=tickers).fillna(0.0)
    daily_invested = pd.Series(index=all_dates, dtype=float).fillna(0.0)
    
    portfolio = {t: {'qty': 0.0, 'total_cost_krw': 0.0, 'avg_cost_krw': 0.0} for t in tickers}

    for date in all_dates:
        today_events = df_all_events[df_all_events['Date'].dt.date == date.date()]
        
        for _, row in today_events.iterrows():
            ticker = row['Ticker']
            t_type = row['Type']
            qty = row['Qty']
            price = row['Price']
            
            fx_rate = fx_data.asof(row['Date']) if not pd.isna(fx_data.asof(row['Date'])) else 1300.0
            exchange_multiplier = 1 if is_korean_stock(ticker) else fx_rate
            price_krw = price * exchange_multiplier
            
            pos = portfolio[ticker]
            
            if t_type == 'Split':
                if pos['qty'] > 0:
                    pos['qty'] *= qty
                    pos['avg_cost_krw'] /= qty 
            elif t_type == 'Buy':
                pos['total_cost_krw'] += qty * price_krw
                pos['qty'] += qty
                if pos['qty'] > 0:
                    pos['avg_cost_krw'] = pos['total_cost_krw'] / pos['qty'] 
            elif t_type == 'Sell':
                if pos['qty'] > 0:
                    pos['total_cost_krw'] -= qty * pos['avg_cost_krw'] 
                    pos['qty'] -= qty
                    if pos['qty'] < 1e-4:
                        pos['qty'] = 0.0
                        pos['avg_cost_krw'] = 0.0
                        pos['total_cost_krw'] = 0.0
                        
        current_day_invested = 0.0
        for t in tickers:
            daily_qty.at[date, t] = portfolio[t]['qty']
            current_day_invested += portfolio[t]['total_cost_krw'] 
            
        daily_invested[date] = current_day_invested

    daily_valuation_krw = pd.DataFrame(index=all_dates)
    
    for col in daily_qty.columns:
        if col not in market_data.columns:
            st.error(f"⚠️ '{col}' 종목의 주가 데이터를 찾을 수 없습니다.")
            daily_valuation_krw[col] = 0
            continue 
            
        exchange_multiplier = 1 if is_korean_stock(col) else fx_data 
        daily_valuation_krw[col] = daily_qty[col] * market_data[col] * exchange_multiplier
            
    daily_total_asset = daily_valuation_krw.sum(axis=1)
    
    result_df = pd.DataFrame({
        'Total_Asset': daily_total_asset,
        'Invested_Principal': daily_invested
    })
    
    result_df['Return_Rate'] = ((result_df['Total_Asset'] - result_df['Invested_Principal']) / result_df['Invested_Principal'] * 100).fillna(0)

    # --- 상단 요약 지표 ---
    current_asset = result_df['Total_Asset'].iloc[-1]
    current_invested = result_df['Invested_Principal'].iloc[-1]
    total_return_pct = result_df['Return_Rate'].iloc[-1]
    total_profit = current_asset - current_invested
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 평가 금액", f"{current_asset:,.0f} 원")
    col2.metric("총 투자 원금", f"{current_invested:,.0f} 원")
    col3.metric("누적 수익률", f"{total_return_pct:.2f}%", f"{total_profit:,.0f} 원")
    
    st.divider()

    st.subheader("📊 일자별 자산 및 수익률 추이")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Total_Asset'], fill='tozeroy', name="총 평가 자산 (원)", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Invested_Principal'], name="투자 원금 (원)", line=dict(color='darkorange', width=2, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Return_Rate'], name="수익률 (%)", line=dict(color='firebrick', width=2, dash='dot')), secondary_y=True)
    
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="금액 (원)", secondary_y=False)
    fig.update_yaxes(title_text="수익률 (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # (기존 코드) st.plotly_chart(fig, use_container_width=True) 바로 아래에 추가하세요.
    
    # ====================================================================
    # --- 6. 연도별 투자 성과 요약 ---
    # ==========================================
    st.divider()
    st.subheader("📅 연도별 투자 성과 요약")
    
    # 1. 매년 마지막 거래일의 데이터만 추출
    result_df_copy = result_df.copy()
    result_df_copy['Year'] = result_df_copy.index.year
    df_yearly = result_df_copy.drop_duplicates(subset=['Year'], keep='last').copy()
    df_yearly.set_index('Year', inplace=True)
    
    # 2. 전년도 말 기준 자산 및 원금 세팅 (계산용)
    df_yearly['Prev_Asset'] = df_yearly['Total_Asset'].shift(1).fillna(0)
    df_yearly['Prev_Invested'] = df_yearly['Invested_Principal'].shift(1).fillna(0)
    
    # 3. 해당 연도의 순수익 및 수익률 계산
    df_yearly['Net_Invested_This_Year'] = df_yearly['Invested_Principal'] - df_yearly['Prev_Invested']
    df_yearly['Profit_This_Year'] = df_yearly['Total_Asset'] - df_yearly['Prev_Asset'] - df_yearly['Net_Invested_This_Year']
    
    # 연간 수익률 = 당해 연도 순수익금 / (작년 말 자산 + 올해 순투자액) * 100
    base_capital = df_yearly['Prev_Asset'] + df_yearly['Net_Invested_This_Year']
    
    # 분모가 0인 경우(투자금 0) 에러 방지
    df_yearly['Annual_Return_Rate'] = 0.0
    valid_idx = base_capital > 0
    df_yearly.loc[valid_idx, 'Annual_Return_Rate'] = (df_yearly.loc[valid_idx, 'Profit_This_Year'] / base_capital.loc[valid_idx]) * 100

    # 4. 출력용 데이터 프레임 정리
    display_yearly = pd.DataFrame({
        '연말 평가 자산(KRW)': df_yearly['Total_Asset'],
        '연간 순투자액(KRW)': df_yearly['Net_Invested_This_Year'],
        '연간 순수익금(KRW)': df_yearly['Profit_This_Year'],
        '연간 수익률(%)': df_yearly['Annual_Return_Rate'],
        '누적 수익률(%)': df_yearly['Return_Rate']
    })
    
    styled_yearly = display_yearly.style.format({
        '연말 평가 자산(KRW)': '{:,.0f} 원',
        '연간 순투자액(KRW)': '{:,.0f} 원',
        '연간 순수익금(KRW)': '{:,.0f} 원',
        '연간 수익률(%)': '{:,.2f}%',
        '누적 수익률(%)': '{:,.2f}%'
    })
    
    # 표 상단에 간단한 설명 추가
    st.markdown("매년 말 기준의 자산 현황과 **해당 연도에 새롭게 발생한 수익 및 수익률**을 보여줍니다.")
    st.dataframe(styled_yearly, use_container_width=True)
    # ====================================================================


    st.divider()
    st.subheader("⏱️ 타임머신: 특정 일자의 포트폴리오 엿보기")
    
    min_date = start_date.date()
    max_date = end_date.date()
    
    selected_date = st.slider(
        "날짜를 드래그하여 과거의 포트폴리오 상태를 확인하세요.",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD"
    )
    
    df_snapshot = get_portfolio_snapshot(df_trade, selected_date, fx_data, name_mapping, market_data)
    
    col_pie, col_summary = st.columns([1, 1])
    
    with col_pie:
        st.markdown(f"**🥧 {selected_date} 기준 비중**")
        if not df_snapshot.empty:
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_snapshot['종목명'], 
                values=df_snapshot['보유 금액(KRW)'], 
                hole=.3,
                textinfo='label+percent'
            )])
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("선택하신 일자에는 보유 중인 주식이 없습니다.")

    with col_summary:
        st.markdown(f"**💼 {selected_date} 기준 요약**")
        if not df_snapshot.empty:
            total_eval = df_snapshot['보유 금액(KRW)'].sum()
            total_invest = (df_snapshot['보유 금액(KRW)'] - df_snapshot['수익금(KRW)']).sum()
            total_profit = df_snapshot['수익금(KRW)'].sum()
            total_rate = (total_profit / total_invest * 100) if total_invest > 0 else 0
            
            st.metric("해당일 총 평가 금액", f"{total_eval:,.0f} 원")
            st.metric("해당일 투자 원금", f"{total_invest:,.0f} 원")
            st.metric("해당일 합계 수익률", f"{total_rate:.2f}%", f"{total_profit:,.0f} 원")
    
    st.markdown(f"**📋 {selected_date} 기준 상세 현황**")
    if not df_snapshot.empty:
        styled_snapshot_df = df_snapshot.style.format({
            '매수 단가(KRW)': '{:,.0f} 원',
            '보유 수량': '{:,.2f}',
            '보유 금액(KRW)': '{:,.0f} 원',
            '수익금(KRW)': '{:,.0f} 원',
            '수익률 (%)': '{:,.2f}%'
        })
        st.dataframe(styled_snapshot_df, use_container_width=True)

    st.divider()
    st.subheader("💰 종목별 실현 수익 (매도 내역)")
    
    df_realized = calculate_realized_profit(df_trade, fx_data, name_mapping)
    
    if not df_realized.empty:
        styled_df = df_realized.style.format({
            '매도 단가(KRW)': '{:,.0f} 원',
            '매수 평단가(KRW)': '{:,.0f} 원',
            '실현 수익금(KRW)': '{:,.0f} 원',
            '수익률 (%)': '{:,.2f}%'
        })
        st.dataframe(styled_df, use_container_width=True)
        
        total_realized = df_realized['실현 수익금(KRW)'].sum()
        if total_realized >= 0:
            st.success(f"🎉 누적 실현 수익금: **{total_realized:,.0f} 원**")
        else:
            st.error(f"📉 누적 실현 손실금: **{total_realized:,.0f} 원**")
    else:
        st.info("아직 매도(수익 실현) 내역이 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 매매 이력 CSV 파일을 업로드해 주세요.")
