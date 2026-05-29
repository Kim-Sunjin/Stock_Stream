import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# ==========================================
# 1. 맵핑 및 설정 정보
# ==========================================
TICKER_MAP = {
    # 미국 주식
    "엔비디아": "NVDA", "애플": "AAPL", "알파벳 Class A": "GOOGL",
    "테슬라": "TSLA", "아마존닷컴": "AMZN", "마이크로소프트": "MSFT",
    "버크셔 해서웨이 Class B": "BRK-B", "뱅크오브아메리카": "BAC",
    "팔란티어 테크놀로지스": "PLTR", "비자": "V", "코카콜라": "KO",
    "포드 모터": "F", "Schwab 미국 배당주 ETF": "SCHD",
    "Invesco 미국 나스닥 100 ETF": "QQQM", "AMD(어드밴스드 마이크로 디바이시스)": "AMD",
    "어플라이드 머티어리얼즈": "AMAT", "ASML 홀딩 ADR": "ASML",
    "블랙 다이아몬드 테라퓨틱스": "BDTX", "넥스트에라 에너지": "NEE",
    "Vanguard Intermediate-Term Government Bo": "VGIT",
    
    # 국내 주식
    "삼성전자": "005930.KS", "삼성전자우": "005935.KS", "현대차": "005380.KS",
    "기아": "000270.KS", "NAVER": "035420.KS", "카카오": "035720.KS",
    "삼성SDI": "006400.KS", "LG화학": "051910.KS", "LG에너지솔루션": "373220.KS",
    "한화에어로스페이스": "012450.KS", "HD현대일렉트릭": "267260.KS",
    "신성이엔지": "011930.KS", "에코프로비엠": "247540.KQ", "파마리서치": "214450.KQ",
    "TIGER 미국S&P500": "360750.KS", "케이사인": "192250.KQ", "KODEX 미국나스닥AI테크액티브": "473490.KS",
    "KODEX 미국S&P500": "360200.KS", "ACE 미국30년국채액티브": "453850.KS"
}

# 2. 기업 이벤트 정밀 보정 (액면병합/분할 등 시세 왜곡 방지)
CORPORATE_EVENTS = {
    "신성이엔지": {"ratio": 0.1, "effective_date": "2026-05-15"},
    "케이사인": {"ratio": 0.1, "effective_date": "2024-11-01"}, 
    "카카오": {"ratio": 5.0, "effective_date": "2021-04-15"},
}

def is_korean_stock(ticker):
    return str(ticker).endswith('.KS') or str(ticker).endswith('.KQ')

# ==========================================
# 2. 데이터 처리 함수
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # 비주식 자산(환전, 채권 등) 강제 필터링
    df = df[~df['거래명'].astype(str).str.contains('외화매수|외화매도', na=False)]
    df = df[~df['종목명'].astype(str).str.contains('국고|채권', na=False)]
    
    df['Date'] = pd.to_datetime(df['거래일자'], errors='coerce')
    df['종목명'] = df['종목명'].astype(str).str.replace('USD ', '', regex=False).str.strip()
    df['Ticker'] = df['종목명'].map(TICKER_MAP)
    
    for col in ['거래수량', '거래단가', '거래번호']:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    df['Type'] = df['거래명'].apply(lambda x: 'Buy' if pd.notna(x) and '매수' in str(x) else ('Sell' if pd.notna(x) and '매도' in str(x) else 'Other'))
    df['Qty'] = df['거래수량']
    df['Price'] = df['거래단가']
    
    # 기업 이벤트 데이터 보정 (수익률 왜곡 해결의 핵심)
    for name, event in CORPORATE_EVENTS.items():
        mask = (df['종목명'] == name) & (df['Date'] < pd.to_datetime(event['effective_date']))
        if mask.any():
            df.loc[mask, 'Qty'] = df.loc[mask, 'Qty'] * event['ratio']
            df.loc[mask, 'Price'] = df.loc[mask, 'Price'] / event['ratio']

    # 같은 날짜 내에서 과거의 거래부터 역순으로 정렬되도록 조정
    if '거래번호' in df.columns:
        df = df.sort_values(by=['Date', '거래번호'], ascending=[True, False]).reset_index(drop=True)
    else:
        df['SortOrder'] = df['Type'].map({'Buy': 1, 'Sell': 2, 'Other': 3})
        df = df.sort_values(by=['Date', 'SortOrder'], ascending=[True, True]).reset_index(drop=True)
        
    return df[df['Type'].isin(['Buy', 'Sell'])]

@st.cache_data(ttl=3600)
def load_market_data(tickers, start_date):
    end_date = pd.Timestamp.today().normalize()
    valid_tickers = [t for t in tickers if pd.notna(t)]
    market_data = pd.DataFrame()
    fx_data = pd.Series(dtype=float)
    
    if valid_tickers:
        try:
            data = yf.download(valid_tickers, start=start_date, end=end_date)['Close']
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            market_data = data if len(valid_tickers) > 1 else pd.DataFrame({valid_tickers[0]: data})
        except Exception:
            pass
            
    try:
        usdkrw = yf.download("KRW=X", start=start_date, end=end_date)['Close']
        fx_data = usdkrw if isinstance(usdkrw, pd.Series) else usdkrw.iloc[:, 0]
    except Exception:
        pass
        
    return market_data, fx_data

# ==========================================
# 3. UI 렌더링 및 메인 로직
# ==========================================
st.set_page_config(page_title="주식 종합 포트폴리오 분석", layout="wide")
st.title("📈 통합 주식 포트폴리오 대시보드")

st.sidebar.markdown("**[작성 규칙]**\n- 한국 주식: 원화(KRW) 단가 기록\n- 미국 주식: 달러(USD) 단가 기록")
uploaded_file = st.sidebar.file_uploader("매매 이력 CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    df_trade = load_and_clean_data(uploaded_file)
    tickers = df_trade['Ticker'].dropna().unique().tolist()
    start_date = df_trade['Date'].min()
    end_date = pd.Timestamp.today().normalize()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    st.sidebar.success("데이터 로드 및 이벤트 보정 완료!")
    
    with st.spinner("종목명, 주가 및 환율 데이터를 실시간으로 불러오는 중입니다..."):
        market_data_raw, fx_data_raw = load_market_data(tickers, start_date)
        market_data = market_data_raw.copy().reindex(all_dates).ffill().bfill()
        fx_data = fx_data_raw.copy().reindex(all_dates).ffill().bfill()

    # ==========================================
    # --- 🔍 개별 종목 기술적 분석 섹션 ---
    # ==========================================
    st.divider()
    st.subheader("🔍 보유 종목 기술적 분석 (RSI, 볼린저 밴드, 이동평균선)")
    
    valid_names = df_trade[df_trade['Ticker'].notna()]['종목명'].unique().tolist()
    selected_name = st.sidebar.selectbox("기술적 분석 종목 선택", options=valid_names)
    selected_ticker = TICKER_MAP.get(selected_name)
    
    ohlc_data = yf.download(selected_ticker, start=start_date, progress=False).copy()
    if isinstance(ohlc_data.columns, pd.MultiIndex):
        ohlc_data.columns = ohlc_data.columns.droplevel(1)
        
    if not ohlc_data.empty:
        ohlc_data['SMA20'] = ohlc_data['Close'].rolling(window=20).mean()
        ohlc_data['SMA60'] = ohlc_data['Close'].rolling(window=60).mean()
        ohlc_data['STD20'] = ohlc_data['Close'].rolling(window=20).std()
        ohlc_data['BB_Upper'] = ohlc_data['SMA20'] + (ohlc_data['STD20'] * 2)
        ohlc_data['BB_Lower'] = ohlc_data['SMA20'] - (ohlc_data['STD20'] * 2)
        
        delta = ohlc_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        ohlc_data['RSI'] = 100 - (100 / (1 + rs))

        st.markdown(f"**[{selected_name}] 일봉 차트 및 지표**")
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig_tech.add_trace(go.Candlestick(
            x=ohlc_data.index, open=ohlc_data['Open'], high=ohlc_data['High'],
            low=ohlc_data['Low'], close=ohlc_data['Close'], name='Price'
        ), row=1, col=1)
        
        fig_tech.add_trace(go.Scatter(x=ohlc_data.index, y=ohlc_data['BB_Upper'], name='BB 상단', line=dict(color='rgba(255, 165, 0, 0.6)', width=1)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=ohlc_data.index, y=ohlc_data['BB_Lower'], name='BB 하단', line=dict(color='rgba(255, 165, 0, 0.6)', width=1), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.1)'), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=ohlc_data.index, y=ohlc_data['SMA20'], name='20일선', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=ohlc_data.index, y=ohlc_data['SMA60'], name='60일선', line=dict(color='green', width=1.5)), row=1, col=1)
        
        fig_tech.add_trace(go.Scatter(x=ohlc_data.index, y=ohlc_data['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig_tech.add_hline(y=70, line=dict(color='red', width=1, dash='dot'), row=2, col=1)
        fig_tech.add_hline(y=30, line=dict(color='green', width=1, dash='dot'), row=2, col=1)
        
        fig_tech.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified", height=500, margin=dict(l=0, r=0, t=30, b=0))
        fig_tech.update_yaxes(title_text="주가", row=1, col=1)
        fig_tech.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)
    else:
        st.warning(f"'{selected_name}'의 차트 데이터를 불러오지 못했습니다.")

    # ==========================================
    # --- 포트폴리오 통합(KRW 환산) 시뮬레이션 ---
    # ==========================================
    daily_qty = pd.DataFrame(index=all_dates, columns=tickers).fillna(0.0)
    daily_invested = pd.Series(index=all_dates, dtype=float).fillna(0.0)
    
    portfolio = {t: {'qty': 0.0, 'total_cost_krw': 0.0, 'avg_cost_krw': 0.0} for t in tickers}
    realized_profits = []

    for date in all_dates:
        today_events = df_trade[df_trade['Date'].dt.date == date.date()]
        
        for _, row in today_events.iterrows():
            ticker = row['Ticker']
            if pd.isna(ticker): continue
                
            t_type = row['Type']
            qty = row['Qty']
            price = row['Price']
            
            fx_rate = fx_data.asof(row['Date']) if not pd.isna(fx_data.asof(row['Date'])) else 1300.0
            exchange_multiplier = 1 if is_korean_stock(ticker) else fx_rate
            price_krw = price * exchange_multiplier
            
            pos = portfolio[ticker]
            
            if t_type == 'Buy':
                pos['total_cost_krw'] += qty * price_krw
                pos['qty'] += qty
                if pos['qty'] > 0:
                    pos['avg_cost_krw'] = pos['total_cost_krw'] / pos['qty'] 
            elif t_type == 'Sell':
                if pos['qty'] > 0:
                    sell_qty = min(qty, pos['qty'])
                    realized_profit = (price_krw - pos['avg_cost_krw']) * sell_qty
                    realized_profits.append({
                        'Date': row['Date'], '종목명': row['종목명'], '매도 수량': sell_qty,
                        '매수 평단가(KRW)': pos['avg_cost_krw'], '매도 단가(KRW)': price_krw,
                        '실현 수익금(KRW)': realized_profit
                    })
                    pos['total_cost_krw'] -= sell_qty * pos['avg_cost_krw'] 
                    pos['qty'] -= sell_qty
                    if pos['qty'] < 1e-4:
                        pos['qty'] = 0.0; pos['avg_cost_krw'] = 0.0; pos['total_cost_krw'] = 0.0
                        
        current_day_invested = 0.0
        for t in tickers:
            daily_qty.at[date, t] = portfolio[t]['qty']
            current_day_invested += portfolio[t]['total_cost_krw'] 
            
        daily_invested[date] = current_day_invested

    # 시세 기반 매일의 자산 가치 평가
    daily_valuation_krw = pd.DataFrame(index=all_dates)
    for col in daily_qty.columns:
        if col in market_data.columns:
            exchange_multiplier = 1 if is_korean_stock(col) else fx_data 
            daily_valuation_krw[col] = daily_qty[col] * market_data[col] * exchange_multiplier
        else:
            daily_valuation_krw[col] = 0
            
    daily_total_asset = daily_valuation_krw.sum(axis=1)
    result_df = pd.DataFrame({'Total_Asset': daily_total_asset, 'Invested_Principal': daily_invested})
    result_df['Return_Rate'] = ((result_df['Total_Asset'] - result_df['Invested_Principal']) / result_df['Invested_Principal'] * 100).fillna(0)

    # ==========================================
    # --- 상단 요약 지표 및 메인 차트 ---
    # ==========================================
    st.divider()
    st.subheader("💼 전체 포트폴리오 요약 (원화 통합 환산)")
    current_asset = result_df['Total_Asset'].iloc[-1]
    current_invested = result_df['Invested_Principal'].iloc[-1]
    total_return_pct = result_df['Return_Rate'].iloc[-1]
    total_profit = current_asset - current_invested
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 평가 금액", f"{current_asset:,.0f} 원")
    col2.metric("총 투자 원금", f"{current_invested:,.0f} 원")
    col3.metric("현재 포트폴리오 수익률", f"{total_return_pct:.2f}%", f"{total_profit:,.0f} 원")
    
    st.divider()
    st.subheader("📊 일자별 자산 및 수익률 추이")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Total_Asset'], fill='tozeroy', name="총 평가 자산 (원)", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Invested_Principal'], name="투자 원금 (원)", line=dict(color='darkorange', width=2, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Return_Rate'], name="수익률 (%)", line=dict(color='firebrick', width=2, dash='dot')), secondary_y=True)
    
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=30, b=0))
    fig.update_yaxes(title_text="금액 (원)", secondary_y=False)
    fig.update_yaxes(title_text="수익률 (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # --- 연도별 투자 성과 요약 ---
    # ==========================================
    st.divider()
    st.subheader("📅 연도별 투자 성과 요약")
    
    result_df_copy = result_df.copy()
    result_df_copy['Year'] = result_df_copy.index.year
    df_yearly = result_df_copy.drop_duplicates(subset=['Year'], keep='last').copy()
    df_yearly.set_index('Year', inplace=True)
    
    df_yearly['Prev_Asset'] = df_yearly['Total_Asset'].shift(1).fillna(0)
    df_yearly['Prev_Invested'] = df_yearly['Invested_Principal'].shift(1).fillna(0)
    
    df_yearly['Net_Invested_This_Year'] = df_yearly['Invested_Principal'] - df_yearly['Prev_Invested']
    df_yearly['Profit_This_Year'] = df_yearly['Total_Asset'] - df_yearly['Prev_Asset'] - df_yearly['Net_Invested_This_Year']
    
    base_capital = df_yearly['Prev_Asset'] + df_yearly['Net_Invested_This_Year']
    df_yearly['Annual_Return_Rate'] = 0.0
    valid_idx = base_capital > 0
    df_yearly.loc[valid_idx, 'Annual_Return_Rate'] = (df_yearly.loc[valid_idx, 'Profit_This_Year'] / base_capital.loc[valid_idx]) * 100

    display_yearly = pd.DataFrame({
        '연말 평가 자산(KRW)': df_yearly['Total_Asset'],
        '연간 순투자액(KRW)': df_yearly['Net_Invested_This_Year'],
        '연간 평가수익금(KRW)': df_yearly['Profit_This_Year'],
        '연간 수익률(%)': df_yearly['Annual_Return_Rate'],
        '누적 수익률(%)': df_yearly['Return_Rate']
    })
    
    st.dataframe(display_yearly.style.format({
        '연말 평가 자산(KRW)': '{:,.0f} 원', '연간 순투자액(KRW)': '{:,.0f} 원',
        '연간 평가수익금(KRW)': '{:,.0f} 원', '연간 수익률(%)': '{:,.2f}%', '누적 수익률(%)': '{:,.2f}%'
    }), use_container_width=True)
    
    # ==========================================
    # --- 타임머신 및 보유 현황 ---
    # ==========================================
    st.divider()
    st.subheader("⏱️ 타임머신: 특정 일자의 포트폴리오 엿보기")
    
    selected_date = st.slider("날짜를 드래그하여 과거의 포트폴리오 상태를 확인하세요.", 
                              min_value=start_date.date(), max_value=end_date.date(), value=end_date.date(), format="YYYY-MM-DD")
    
    snapshot_date = pd.to_datetime(selected_date)
    snapshot_data = []
    
    for tk in tickers:
        qty = daily_qty.loc[snapshot_date, tk]
        if qty > 1e-4:
            name = df_trade[df_trade['Ticker'] == tk]['종목명'].iloc[0]
            
            # 정확한 시뮬레이션 복원
            past_tx = df_trade[(df_trade['Ticker'] == tk) & (df_trade['Date'].dt.date <= snapshot_date.date())]
            tmp_qty = 0.0; tmp_cost = 0.0
            for _, r in past_tx.iterrows():
                ex_rate = fx_data.asof(r['Date']) if not pd.isna(fx_data.asof(r['Date'])) else 1300.0
                p_krw = r['Price'] * (1 if is_korean_stock(tk) else ex_rate)
                if r['Type'] == 'Buy':
                    tmp_cost = (tmp_qty * tmp_cost + r['Qty'] * p_krw) / (tmp_qty + r['Qty'])
                    tmp_qty += r['Qty']
                elif r['Type'] == 'Sell':
                    tmp_qty -= r['Qty']
                    if tmp_qty < 1e-4: tmp_cost = 0.0; tmp_qty = 0.0
            avg_cost = tmp_cost
            
            curr_p = market_data.loc[snapshot_date, tk] if tk in market_data.columns else 0
            ex_rate = fx_data.loc[snapshot_date]
            curr_p_krw = curr_p * (1 if is_korean_stock(tk) else ex_rate)
            
            eval_amount = qty * curr_p_krw
            profit = eval_amount - (qty * avg_cost)
            rate = (profit / (qty * avg_cost)) * 100 if avg_cost > 0 else 0
            
            snapshot_data.append({
                '종목명': name, '매수 단가(KRW)': avg_cost, '현재 단가(KRW)': curr_p_krw,
                '보유 수량': qty, '보유 금액(KRW)': eval_amount, '수익금(KRW)': profit, '수익률 (%)': rate
            })
            
    df_snapshot = pd.DataFrame(snapshot_data)
    
    col_pie, col_summary = st.columns([1, 1])
    with col_pie:
        st.markdown(f"**🥧 {selected_date} 기준 비중**")
        if not df_snapshot.empty:
            fig_pie = go.Figure(data=[go.Pie(labels=df_snapshot['종목명'], values=df_snapshot['보유 금액(KRW)'], hole=.3, textinfo='label+percent')])
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
    def style_returns(val):
        if pd.isna(val): return ''
        if isinstance(val, str):
            try: val = float(val.replace(',', '').replace('$', '').replace('%', ''))
            except ValueError: return ''
        color = 'red' if val > 0 else 'blue' if val < 0 else 'black'
        return f'color: {color}'
        
    if not df_snapshot.empty:
        st.dataframe(df_snapshot.style.format({
            '매수 단가(KRW)': '{:,.0f} 원', '현재 단가(KRW)': '{:,.0f} 원', '보유 수량': '{:,.2f}', 
            '보유 금액(KRW)': '{:,.0f} 원', '수익금(KRW)': '{:,.0f} 원', '수익률 (%)': '{:,.2f}%'
        }).map(style_returns, subset=['수익금(KRW)', '수익률 (%)']), use_container_width=True)

    # ==========================================
    # --- 종목별 실현 수익 ---
    # ==========================================
    st.divider()
    st.subheader("💰 종목별 누적 실현 수익 (매도 내역)")
    
    df_realized = pd.DataFrame(realized_profits)
    if not df_realized.empty:
        df_realized_summary = df_realized.groupby('종목명').apply(lambda x: pd.Series({
            '총 매도 수량': x['매도 수량'].sum(),
            '평균 매수 단가(KRW)': (x['매수 평단가(KRW)'] * x['매도 수량']).sum() / x['매도 수량'].sum(),
            '평균 매도 단가(KRW)': (x['매도 단가(KRW)'] * x['매도 수량']).sum() / x['매도 수량'].sum(),
            '총 실현 수익금(KRW)': x['실현 수익금(KRW)'].sum()
        })).reset_index()
        
        df_realized_summary['수익률 (%)'] = (df_realized_summary['평균 매도 단가(KRW)'] - df_realized_summary['평균 매수 단가(KRW)']) / df_realized_summary['평균 매수 단가(KRW)'] * 100
        df_realized_summary = df_realized_summary.sort_values(by='총 실현 수익금(KRW)', ascending=False).reset_index(drop=True)
        
        st.dataframe(df_realized_summary.style.format({
            '총 매도 수량': '{:,.2f}', '평균 매수 단가(KRW)': '{:,.0f} 원', '평균 매도 단가(KRW)': '{:,.0f} 원',
            '총 실현 수익금(KRW)': '{:,.0f} 원', '수익률 (%)': '{:,.2f}%'
        }).map(style_returns, subset=['총 실현 수익금(KRW)', '수익률 (%)']), use_container_width=True)
        
        total_real = df_realized_summary['총 실현 수익금(KRW)'].sum()
        if total_real >= 0:
            st.success(f"🎉 누적 총 실현 수익금: **{total_real:,.0f} 원**")
        else:
            st.error(f"📉 누적 총 실현 손실금: **{total_real:,.0f} 원**")
    else:
        st.info("아직 매도(수익 실현) 내역이 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 매매 이력 CSV 파일을 업로드해 주세요.")
