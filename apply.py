import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 나의 주식 포트폴리오 대시보드 (원/달러 통합형)")

# --- [함수] 한국 주식 완벽 식별 ---
def is_korean_stock(ticker):
    t = str(ticker).strip().upper()
    return t.endswith(('.KS', '.KQ', '.KR')) or t.replace('.KS', '').replace('.KQ', '').isdigit()

# --- [함수] 티커를 종목명으로 변환 ---
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

# --- [함수] API 액면분할 이벤트 수집 (주가 복원 전용) ---
@st.cache_data(ttl=86400)
def get_split_events(tickers, start_date):
    split_records = []
    for ticker in tickers:
        try:
            # 야후 파이낸스는 한국 주식에 .KS가 필요하므로 검색 시에만 붙여줌
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
                            'Ticker': ticker, # 원래 티커 이름으로 저장
                            'Ratio': ratio    # 수량이 아니라 순수 '비율'로 저장
                        })
        except Exception:
            continue
    return pd.DataFrame(split_records) 

# --- [함수] 타임머신: 특정 일자 기준 포트폴리오 스냅샷 ---
def get_portfolio_snapshot(df_trades, target_date, fx_series, name_mapping, market_data, df_splits):
    target_date_pd = pd.to_datetime(target_date)
    
    if not df_splits.empty:
        df_combined = pd.concat([df_trades, df_splits], ignore_index=True)
    else:
        df_combined = df_trades.copy()
        
    df_combined = df_combined[df_combined['Date'] <= target_date_pd].copy()
    df_combined['SortOrder'] = df_combined['Type'].map({'Split': 1, 'Buy': 2, 'Sell': 3})
    df = df_combined.sort_values(['Date', 'SortOrder']).copy()
    
    portfolio = {} 
    
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        t_type = row['Type']
        qty = row['Qty']
        price = row['Price']
        date = row['Date']
        
        # [복구됨] 거래 당시의 환율을 적용하여 원화 단가로 변환합니다.
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

# --- [함수] 실현 수익 계산 ---
def calculate_realized_profit(df_trades, fx_series, name_mapping, df_splits):
    if not df_splits.empty:
        df_combined = pd.concat([df_trades, df_splits], ignore_index=True)
    else:
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
        
        # [복구됨] 거래 당시의 환율을 적용하여 원화 단가로 변환합니다.
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
# 메인 어플리케이션 실행부
# ==========================================
st.sidebar.markdown("**[작성 규칙]**\n- 한국 주식: 원화(KRW) 입력\n- 미국 주식: 달러(USD) 입력")
uploaded_file = st.sidebar.file_uploader("매매 이력 CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    df_trade = pd.read_csv(uploaded_file)
    
    # 데이터 정제 (띄어쓰기, 대소문자 오타 완벽 방어)
    df_trade['Date'] = pd.to_datetime(df_trade['Date'])
    df_trade['Qty'] = pd.to_numeric(df_trade['Qty'].astype(str).str.replace(',', '').str.strip())
    df_trade['Price'] = pd.to_numeric(df_trade['Price'].astype(str).str.replace(',', '').str.strip())
    df_trade['Type'] = df_trade['Type'].astype(str).str.strip().str.capitalize()
    
    tickers = df_trade['Ticker'].unique().tolist()
    start_date = df_trade['Date'].min()
    end_date = pd.Timestamp.today().normalize()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    st.sidebar.success("데이터 로드 완료!")
    
    @st.cache_data(ttl=3600)
    def load_market_data(tickers, start):
        query_tickers = [t + '.KS' if str(t).isdigit() else t for t in tickers]
        stock_data = yf.download(query_tickers, start=start, progress=False)['Close']
        
        if isinstance(stock_data, pd.Series):
            stock_data = stock_data.to_frame(name=tickers[0])
        else:
            # [버그 수정] 알파벳 순서로 섞인 컬럼을 정확히 1:1 매칭하여 원래 이름으로 복구
            rename_dict = {q: t for q, t in zip(query_tickers, tickers)}
            stock_data = stock_data.rename(columns=rename_dict)
            
        fx_data = yf.download('KRW=X', start=start, progress=False)['Close']
        if isinstance(fx_data, pd.DataFrame):
            fx_data = fx_data.squeeze()
            
        # 시차(Timezone) 에러 원천 차단
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)
        if fx_data.index.tz is not None:
            fx_data.index = fx_data.index.tz_localize(None)
            
        return stock_data, fx_data
        
    with st.spinner("종목명, 주가, 환율 및 분할 데이터를 불러오는 중입니다..."):
        name_mapping = get_company_names(tickers)
        market_data, fx_data = load_market_data(tickers, start_date)
        df_splits = get_split_events(tickers, start_date) 
        
    market_data = market_data.reindex(all_dates).ffill().bfill()
    fx_data = fx_data.reindex(all_dates).ffill().bfill()
    
    # ====================================================================
    # --- [핵심 복원 로직] 야후 API 분할 정보를 이용해 '과거 실제 주가' 완벽 복원 ---
    # ====================================================================
    # 엑셀 기록과 무관하게, API가 알려주는 분할 비율을 역산하여 과거 주가를 되살립니다.
    if not df_splits.empty:
        for _, row in df_splits.iterrows():
            split_date = row['Date']
            ticker = row['Ticker']
            ratio = row['Ratio']
            
            if ticker in market_data.columns:
                # 분할일 '이전'의 모든 주가에 분할 비율을 곱해 깎이기 전 가격으로 복구
                mask = market_data.index < split_date
                market_data.loc[mask, ticker] *= ratio
    # ====================================================================

    # --- 투자 원금 및 수량 시뮬레이션 (API 분할 로직 완전 제거 유지) ---
    df_all_events = df_trade.copy() # 내 엑셀 장부만 100% 신뢰하여 시뮬레이션
    
    # --- 투자 원금 및 수량 시뮬레이션 ---
    manual_splits = df_trade[df_trade['Type'] == 'Split']
    manual_tickers = manual_splits['Ticker'].unique()
    if not df_splits.empty:
        df_splits = df_splits[~df_splits['Ticker'].isin(manual_tickers)]

    df_all_events = pd.concat([df_trade, df_splits], ignore_index=True)
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
            
            # [복구됨] 거래 당시의 환율을 적용하여 원화 단가로 변환합니다.
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

    # --- 5. 일자별 자산 그래프 ---
    st.subheader("📊 일자별 자산 및 수익률 추이")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Total_Asset'], fill='tozeroy', name="총 평가 자산 (원)", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Invested_Principal'], name="투자 원금 (원)", line=dict(color='darkorange', width=2, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Return_Rate'], name="수익률 (%)", line=dict(color='firebrick', width=2, dash='dot')), secondary_y=True)
    
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="금액 (원)", secondary_y=False)
    fig.update_yaxes(title_text="수익률 (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    df_snapshot = get_portfolio_snapshot(df_trade, selected_date, fx_data, name_mapping, market_data, df_splits)
    
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
    
    df_realized = calculate_realized_profit(df_trade, fx_data, name_mapping, df_splits)
    
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