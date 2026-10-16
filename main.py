import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 기온 예측기")
st.write("서울 기온 데이터를 바탕으로 연평균기온의 추세를 분석하고 미래 기온을 예측하는 앱입니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding='utf-8')
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 평균기온과 관측일수 계산
    yearly_stats = df.groupby('연도').agg(
        평균기온=('평균기온', 'mean'),
        관측일수=('날짜', 'count')
    ).reset_index()
    
    # 관측일이 300일 이상인 해만 남김 & 결측치 제거
    valid_yearly = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    valid_yearly['연도'] = valid_yearly['연도'].astype(int)
    valid_yearly = valid_yearly.dropna(subset=['평균기온', '연도'])
    
    return valid_yearly

# 변수명을 교재 스니펫에 맞춰 yearly로 정의
yearly = load_and_preprocess_data()

if yearly.empty:
    st.error("조건을 만족하는 데이터가 없습니다.")
else:
    # 1. 전체 기간 회귀 분석
    x_all = yearly['연도'].astype(float)
    y_all = yearly['평균기온'].astype(float)
    
    a, intercept_all = np.polyfit(x_all, y_all, 1)
    corr_all = np.corrcoef(x_all, y_all)[0, 1]
    
    # 2. 최근 20년 회귀 분석 (제시해주신 스니펫 코드 반영)
    recent = yearly[yearly["연도"] >= yearly["연도"].max() - 19]
    if not recent.empty and len(recent) > 1:
        a2, _ = np.polyfit(recent["연도"].astype(float), recent["평균기온"].astype(float), 1)
    else:
        a2 = 0.0

    # 데이터 메타데이터 추출
    max_year = int(yearly['연도'].max())
    min_year = int(yearly['연도'].min())
    num_years = len(yearly)

    # 화면 정보 출력
    st.subheader("📊 데이터 요약 정보")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("직선을 만든 해의 개수", f"{num_years}개")
    with col_info2:
        st.metric("시작 연도", f"{min_year}년")
    with col_info3:
        st.metric("끝 연도", f"{max_year}년")

    st.markdown("---")

    # 전체 기간 vs 최근 20년 기울기 비교 (나란히 배치)
    st.subheader("📈 기울기 비교 (100년당 기온 변화)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("전체 기간 기울기", f"{a * 100:+.2f}℃ / 100년", help=f"전체 기간 ({min_year}~{max_year}) 기준")
    with c2:
        st.metric("최근 20년 기울기", f"{a2 * 100:+.2f}℃ / 100년", help=f"최근 20년 ({int(recent['연도'].min())}~{max_year}) 기준")

    st.markdown("---")

    # Plotly 시각화 (산점도 및 회귀 직선)
    st.subheader("📉 연도별 연평균기온 산점도 및 추세선")
    
    yearly['회귀예측'] = a * yearly['연도'] + intercept_all
    
    fig = px.scatter(
        yearly, x='연도', y='평균기온', 
        labels={'연도': '연도', '평균기온': '연평균기온 (℃)'},
        title=f"서울 연평균기온 추이 (상관계수: {corr_all:.2f})"
    )
    fig.update_traces(marker=dict(size=8, color='royalblue'))
    
    fig.add_scatter(
        x=yearly['연도'], y=yearly['회귀예측'], 
        mode='lines', name=f'회귀 직선 (100년당 {a * 100:+.2f}℃)',
        line=dict(color='red', width=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 연도 슬라이더를 통한 예상 기온 표시
    st.subheader("🔮 연도별 예상 기온 계산기")
    selected_year = st.slider("조회할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)
    
    predicted_temp = a * selected_year + intercept_all
    
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <h3>{selected_year}년 서울 예상 연평균기온</h3>
            <h1 style="color: #ff4b4b; font-size: 48px;">{predicted_temp:.2f} ℃</h1>
            <p style="color: gray; font-size: 14px;">전체 기간 회귀 분석 모델(100년당 {a * 100:+.2f}℃ 변화) 기준 예측값입니다.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
