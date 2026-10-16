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
    
    # 연도별 평균기온과 관측일수 계산 (결측치 제외)
    yearly_stats = df.groupby('연도').agg(
        평균기온=('평균기온', 'mean'),
        관측일수=('날짜', 'count')
    ).reset_index()
    
    # 관측일이 300일 이상인 해만 남김 & 연도 정수형 변환 및 결측치 제거
    valid_yearly = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
    valid_yearly['연도'] = valid_yearly['연도'].astype(int)
    valid_yearly = valid_yearly.dropna(subset=['평균기온', '연도'])
    
    return valid_yearly

df_yearly = load_and_preprocess_data()

if df_yearly.empty:
    st.error("조건을 만족하는 데이터가 없습니다.")
else:
    # 1. 전체 기간 회귀 분석
    x_all = df_yearly['연도'].astype(float)
    y_all = df_yearly['평균기온'].astype(float)
    
    slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
    corr_all = np.corrcoef(x_all, y_all)[0, 1]
    
    # 100년당 기온 상승 폭 (도 / 100년)
    slope_100_all = slope_all * 100

    # 2. 최근 20년 기간 회귀 분석 (데이터에 존재하는 가장 최근 연도 기준 20년)
    max_year = int(df_yearly['연도'].max())
    min_recent_year = max_year - 19
    df_recent = df_yearly[df_yearly['연도'] >= min_recent_year]
    
    if not df_recent.empty and len(df_recent) > 1:
        x_rec = df_recent['연도'].astype(float)
        y_rec = df_recent['평균기온'].astype(float)
        slope_rec, intercept_rec = np.polyfit(x_rec, y_rec, 1)
        slope_100_rec = slope_rec * 100
    else:
        slope_100_rec = 0.0

    # 화면 정보 출력
    st.subheader("📊 데이터 요약 정보")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("직선을 만든 해의 개수", f"{len(df_yearly)}개")
    with col_info2:
        st.metric("시작 연도", f"{int(df_yearly['연도'].min())}년")
    with col_info3:
        st.metric("끝 연도", f"{max_year}년")

    st.markdown("---")

    # 기울기 비교 섹션
    st.subheader("📈 기울기 비교 (100년당 기온 변화)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.metric("전체 기간 기울기", f"{slope_100_all:+.2f} ℃ / 100년", help=f"전체 기간 ({int(df_yearly['연도'].min())}~{max_year}) 기준")
    with col_g2:
        st.metric("최근 20년 기울기", f"{slope_100_rec:+.2f} ℃ / 100년", help=f"최근 20년 ({min_recent_year}~{max_year}) 기준")

    st.markdown("---")

    # Plotly 시각화 (산점도 및 회귀 직선)
    st.subheader("📉 연도별 연평균기온 산점도 및 추세선")
    
    # 회귀선 예측값 추가
    df_yearly['회귀예측'] = slope_all * df_yearly['연도'] + intercept_all
    
    fig = px.scatter(
        df_yearly, x='연도', y='평균기온', 
        labels={'연도': '연도', '평균기온': '연평균기온 (℃)'},
        title=f"서울 연평균기온 추이 (상관계수: {corr_all:.2f})"
    )
    # 산점도 점 스타일 조정
    fig.update_traces(marker=dict(size=8, color='royalblue'))
    
    # 회귀 직선 추가
    fig.add_scatter(
        x=df_yearly['연도'], y=df_yearly['회귀예측'], 
        mode='lines', name=f'회귀 직선 (100년당 {slope_100_all:+.2f}℃)',
        line=dict(color='red', width=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 연도 슬라이더를 통한 예상 기온 표시
    st.subheader("🔮 연도별 예상 기온 계산기")
    selected_year = st.slider("조회할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)
    
    predicted_temp = slope_all * selected_year + intercept_all
    
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <h3>{selected_year}년 서울 예상 연평균기온</h3>
            <h1 style="color: #ff4b4b; font-size: 48px;">{predicted_temp:.2f} ℃</h1>
            <p style="color: gray; font-size: 14px;">전체 기간 회귀 분석 모델(100년당 {slope_100_all:+.2f}℃ 변화) 기준 예측값입니다.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
