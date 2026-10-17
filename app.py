"""
Streamlit 웹앱: 서울 연평균기온 분석
- 연도별 평균/최고/최저기온 꺾은선 그래프 + 5년 이동평균
- 회귀선(추세) 및 미래 연평균기온 예측 시뮬레이터
- 사이드바 연도 범위 슬라이더와 선택 구간 통계
"""

import streamlit as st
import pandas as pd
import numpy as np

# plotly가 없으면 설치 안내 후 실행 중단
try:
    import plotly.graph_objects as go
except ImportError:
    st.error("plotly가 설치되어 있지 않습니다. 터미널에서 다음 명령을 실행해 주세요: `pip install plotly`")
    st.stop()


st.set_page_config(page_title="서울 연평균 기온 분석", layout="wide", page_icon="🌡️")


# ---------- 1. 데이터 로드 ----------
@st.cache_data
def load_data(path: str = "seoul_temperature.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 날짜 값 앞에 탭 문자가 있어 strip 처리
    df["날짜"] = df["날짜"].str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d")
    df["연도"] = df["날짜"].dt.year

    for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["연도", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    )
    return df


# ---------- 2. 연도별 집계 ----------
@st.cache_data
def make_annual(df: pd.DataFrame) -> pd.DataFrame:
    annual = df.groupby("연도").agg(
        평균기온=("평균기온(℃)", "mean"),
        최고기온=("최고기온(℃)", "mean"),
        최저기온=("최저기온(℃)", "mean"),
    ).reset_index()

    # 5년 이동평균 (중앙 정렬)
    annual["이동평균5"] = annual["평균기온"].rolling(window=5, center=True).mean()

    # 연중 최고/최저: 일최고·일최저의 연 최대/최소
    annual["일최고최대"] = df.groupby("연도")["최고기온(℃)"].max().values
    annual["일최저최소"] = df.groupby("연도")["최저기온(℃)"].min().values

    return annual


df = load_data()
annual = make_annual(df)

# ---------- 3. 회귀선 (선형회귀) ----------
x = annual["연도"].to_numpy(dtype=float)
y = annual["평균기온(℃)"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
annual["회귀선"] = slope * annual["연도"] + intercept


# ---------- 4. 미래 예측 시뮬레이션 ----------
def predict_temp(year: int) -> float:
    return float(slope * year + intercept)


# ---------- 5. UI ----------
st.title("🌡️ 서울 연평균기온 분석")
st.caption("연도별 연평균기온 · 5년 이동평균 · 회귀선(추세) · 미래 예측 시뮬레이터")

with st.sidebar:
    st.header("설정")

    year_min, year_max = int(annual["연도"].min()), int(annual["연도"].max())

    # 연도 범위 슬라이더
    sel_min, sel_max = st.slider(
        "연도 범위",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    # 미래 예측 연도 슬라이더
    pred_year = st.slider(
        "미래 예측 연도",
        min_value=max(2020, year_min),
        max_value=max(2100, year_max + 50),
        value=2050,
    )
    pred_val = predict_temp(pred_year)

    # 선택 구간 통계
    st.subheader("선택 구간 통계 (℃)")
    if sel_max >= sel_min:
        sel = annual[(annual["연도"] >= sel_min) & (annual["연도"] <= sel_max)]
        if not sel.empty:
            mean_avg = sel["평균기온"].mean()
            mean_max = sel["최고기온"].mean()
            mean_min = sel["최저기온"].mean()
            st.metric("연평균기온", f"{mean_avg:.2f}")
            st.metric("연평균 최고기온", f"{mean_max:.2f}")
            st.metric("연평균 최저기온", f"{mean_min:.2f}")
            st.dataframe(
                sel[["연도", "평균기온(℃)", "최고기온(℃)", "최저기온(℃)"]],
                use_container_width=True,
            )

    # 추세 안내
    st.caption(f"회귀 추세: 연평균기온이 10년당 약 {slope * 10:.2f}℃ 상승하는 추세")

# ---------- 6. 그래프 ----------
fig = go.Figure()

# 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온(℃)"],
        mode="lines+markers",
        name="연평균기온",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
    )
)

# 5년 이동평균
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["이동평균5"],
        mode="lines",
        name="5년 이동평균",
        line=dict(color="#d62728", width=2, dash="dash"),
    )
)

# 회귀선
x_line = np.linspace(int(annual["연도"].min()) - 1, 2100, 1000)
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="회귀선(추세)",
        line=dict(color="#2ca02c", width=2, dash="dot"),
    )
)

# 미래 예측 지점
fig.add_trace(
    go.Scatter(
        x=[pred_year],
        y=[pred_val],
        mode="markers",
        name=f"예측 ({pred_year}년)",
        marker=dict(color="red", size=11, symbol="diamond"),
    )
)

fig.update_layout(
    title="서울 연평균기온 추이와 미래 예측",
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    legend=dict(x=0.02, y=0.98),
    hovermode="x unified",
    height=540,
    margin=dict(l=50, r=30, t=60, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- 7. 예측 안내 ----------
st.markdown("---")
st.info("📌 **예측 시뮬레이터**: 회귀선을 바탕으로 한 미래 연평균기온 (실제 관측값이 아닌 추세 기반 추정치입니다)")
st.write(f"선택 연도 **{pred_year}** → 예측 연평균기온 **{pred_val:.2f}℃**")
