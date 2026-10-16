import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 URL
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    """
    서울 기온 데이터를 로드하고 요구사항에 맞게 전처리합니다.
    - 2025년 이하 데이터만 필터링
    - 연도별 데이터 개수(관측일수)가 300일 이상인 해만 추출
    - 연도별 평균기온 계산
    """
    try:
        # UTF-8 인코딩으로 CSV 읽기
        df = pd.read_csv(DATA_URL, encoding='utf-8')
        
        # 날짜 형식 변환 및 연도 추출
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연도'] = df['날짜'].dt.year
        
        # 1. 기준 기간: 2025년까지 필터링
        df_filtered = df[df['연도'] <= 2025].copy()
        
        # 2. 연도별 관측일수 및 평균기온 계산
        yearly_stats = df_filtered.groupby('연도').agg(
            관측일수=('평균기온', 'count'),
            연평균기온=('평균기온', 'mean')
        ).reset_index()
        
        # 3. 관측일이 300일 미만인 해 제외
        yearly_valid = yearly_stats[yearly_stats['관측일수'] >= 300].copy()
        
        return yearly_valid
    except Exception as e:
        st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 데이터 로드
df_yearly = load_and_preprocess_data()

# 메인 제목 및 설명
st.title("🌡️ 서울 기온 예측기 (1900 ~ 2025)")
st.markdown("""
이 앱은 서울의 100년이 넘는 기온 데이터를 바탕으로 **연도별 연평균기온의 추세**를 분석하고, 
회귀 직선을 통해 미래의 기온을 예측해 보는 도구입니다.
""")

if df_yearly.empty:
    st.warning("분석할 유효한 데이터가 없습니다.")
else:
    # 회귀 분석 수행 (연도 -> 연평균기온)
    x = df_yearly['연도']
    y = df_yearly['연평균기온']
    
    # 1차 다항식 피팅 (기울기와 절편 계산)
    slope, intercept = np.polyfit(x, y, 1)
    
    # 상관계수 계산
    correlation = np.corrcoef(x, y)[0, 1]
    
    # 메타데이터 정보 추출
    num_years = len(df_yearly)
    start_year = int(df_yearly['연도'].min())
    end_year = int(df_yearly['연도'].max())
    
    # 사이드바 또는 상단 영역에 메타데이터 표시
    st.sidebar.header("📊 분석 데이터 정보")
    st.sidebar.info(f"""
    - **사용한 해의 개수:** {num_years}개년  
    - **시작 연도:** {start_year}년  
    - **끝 연도:** {end_year}년  
    - **연도와 기온 상관계수:** {correlation:.2f}
    """)
    
    st.markdown("---")
    
    # 레이아웃 구성: 왼쪽(그래프), 오른쪽(예측 도구)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 연도별 연평균기온 산점도 및 추세선")
        
        # Plotly를 이용한 산점도 및 회귀선(추세선) 시각화
        fig = px.scatter(
            df_yearly, 
            x='연도', 
            y='연평균기온',
            labels={'연도': '연도', '연평균기온': '연평균기온 (℃)'},
            title="서울 연도별 연평균기온 변화 추세",
            opacity=0.7
        )
        
        # 회귀선 추가
        df_yearly['회귀예측'] = slope * df_yearly['연도'] + intercept
        fig.add_trace(
            px.line(df_yearly, x='연도', y='회귀예측').data[0]
        )
        # 회귀선 색상 및 이름 조정
        fig.data[1].line.color = 'red'
        fig.data[1].name = '회귀 직선'
        fig.update_traces(showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🔮 기온 예측 시뮬레이터")
        st.markdown("슬라이더를 움직여 원하는 연도의 예상 기온을 확인해 보세요.")
        
        # 1900년부터 2100년까지 선택 가능한 슬라이더
        selected_year = st.slider(
            "예측할 연도를 선택하세요", 
            min_value=1900, 
            max_value=2100, 
            value=2030,
            step=1
        )
        
        # 선택된 연도의 예상 기온 계산
        predicted_temp = slope * selected_year + intercept
        
        # 결과 강조 표시
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-top: 20px;
            ">
                <h3 style="color: #31333F; margin-bottom: 10px;">{selected_year}년 예상 연평균기온</h3>
                <h1 style="color: #ff4b4b; font-size: 42px; margin: 0;">{predicted_temp:.2f} ℃</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.caption(f"💡 회귀 방정식: 연평균기온 = {slope:.4f} × 연도 + {intercept:.2f}")

    # 데이터 요약 테이블 접기 기능
    with st.expander("🔍 전처리된 연도별 데이터 전체 보기"):
        st.dataframe(df_yearly[['연도', '관측일수', '연평균기온']], use_container_width=True)
