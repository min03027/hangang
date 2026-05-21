"""
한강공원 이용객 분석 대시보드
============================
공원별 EDA → t-test → VIF → 모델 학습 → 잔차 도표 → SHAP → Conformal → LSTM
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os
import copy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
from scipy.stats import probplot
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 한글 폰트 설정 (로컬 / Streamlit Cloud 대응)
# ─────────────────────────────────────────────
import platform

sys_name = platform.system()
if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    # Linux / Streamlit Cloud → NanumGothic 시도, 없으면 DejaVu
    try:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "NanumGothic"
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="한강공원 분석 대시보드",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e8ecef;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        color: #666;
        margin: 0 0 0.3rem 0;
        font-weight: 400;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1A1A2E;
    }

    .section-header {
        border-left: 4px solid #2E86AB;
        padding-left: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 700;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown, div[data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 데이터 로드 (캐싱)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """데이터 로드 + 전처리 + 병합"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    df = pd.read_csv(os.path.join(data_dir, "이용객.csv"), encoding="euc-kr")
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M")

    num_cols = [
        "자전거", "인라인", "pm(개인형이동장치)",
        "주요행사", "마라톤", "운동시설", "야구장", "론볼링장",
        "트랙구장", "롤러장", "자전거공원", "외국인",
        "수상시설", "수영장/물놀이장", "빙상장/눈설매장",
        "전망쉼터", "캠핑장", "자연학습장", "음악분수",
        "키즈랜드", "장미원", "x게임장", "자벌레",
        "달빛무지개", "세빛섬", "수상무대", "계절,녹음수광장",
        "천상계단", "피아노물길", "멀티프라자", "서울색공원",
        "물빛광장", "너플들판테라스", "골프장", "여의도샛강",
        "여의도시민 요트나루", "평화공원브릿지", "거울분수",
        "강변물놀이장", "강변프롬나드", "난지 하늘다리",
        "갈대숲탐장로", "꿀벌숲", "치유의숲", "그라스정원",
        "노들섬", "습지생태공원",
    ]

    time_cols = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
    all_num = time_cols + num_cols

    for c in all_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["총이용객"] = df["일반이용자(아침)"] + df["일반이용자(낮)"] + df["일반이용자(저녁)"]

    monthly = df.groupby("연월")[all_num + ["총이용객"]].sum().reset_index()
    monthly["연월"] = monthly["연월"].dt.to_timestamp()

    trend = pd.read_excel(os.path.join(data_dir, "트렌드.xlsx"))
    trend.rename(columns={"날짜": "연월"}, inplace=True)
    trend["연월"] = pd.to_datetime(trend["연월"])

    merged = pd.merge(monthly, trend, on="연월", how="inner")
    merged["검색량"] = merged["한강공원"]

    # 계절 변수
    merged["월"] = merged["연월"].dt.month

    def get_season(m):
        if m in [3, 4, 5]:
            return "봄"
        elif m in [6, 7, 8]:
            return "여름"
        elif m in [9, 10, 11]:
            return "가을"
        return "겨울"

    merged["계절"] = merged["월"].apply(get_season)

    park_list = [
        "광나루한강공원", "이촌한강공원", "뚝섬한강공원", "잠실한강공원",
        "양화한강공원", "망원한강공원", "반포한강공원", "잠원한강공원",
        "강서한강공원", "여의도한강공원", "난지한강공원",
    ]

    return merged, num_cols, all_num, park_list


merged, num_cols, all_num, park_list = load_data()


# ─────────────────────────────────────────────
# 유틸 함수들
# ─────────────────────────────────────────────
def preprocess(X):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X


def stepwise_vif(X_df, threshold=10):
    """VIF가 threshold 초과인 변수를 하나씩 제거 (편향 최소화)"""
    cols = list(X_df.columns)
    removed = []
    while len(cols) > 1:
        X_temp = X_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_c = sm.add_constant(X_temp)
        vifs = []
        for i, col in enumerate(cols):
            try:
                v = variance_inflation_factor(X_c.values, i + 1)
            except Exception:
                v = 0
            vifs.append((col, v))
        vif_df = pd.DataFrame(vifs, columns=["Feature", "VIF"])
        max_row = vif_df.loc[vif_df["VIF"].idxmax()]
        if max_row["VIF"] <= threshold:
            break
        removed.append((max_row["Feature"], round(max_row["VIF"], 1)))
        cols.remove(max_row["Feature"])
    return cols, removed


def get_models():
    return {
        "Ridge": Ridge(alpha=10),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Stacking": StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10)),
                ("gb", GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
                )),
                ("xgb", XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
                )),
            ],
            final_estimator=LinearRegression(),
        ),
    }


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏞️ 분석 설정")
    selected_park = st.selectbox("공원 선택", park_list, index=9)

    st.markdown("---")
    page = st.radio(
        "분석 메뉴",
        [
            "📊 EDA (탐색적 분석)",
            "🔬 t-test & VIF",
            "🤖 모델 학습 & 비교",
            "📈 잔차 도표 & 진단",
            "💡 SHAP 해석",
            "🎯 Conformal Prediction",
            "🧠 LSTM (딥러닝)",
        ],
    )

    st.markdown("---")
    st.markdown(
        f"**데이터 기간**  \n{merged['연월'].min().strftime('%Y-%m')} ~ "
        f"{merged['연월'].max().strftime('%Y-%m')}"
    )
    st.markdown(f"**관측치**: {len(merged)}개월")

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div class="main-header">
    <h1>🏞️ 한강공원 이용객 분석 대시보드</h1>
    <p>현재 선택: <strong>{selected_park}</strong> · 11개 한강공원 비교 분석 · ML/DL 예측</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 공통 변수 준비
# ─────────────────────────────────────────────
leakage = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
feature_cols = [c for c in num_cols if c not in leakage]

# =============================================================
# 📊 EDA
# =============================================================
if page == "📊 EDA (탐색적 분석)":

    # 상단 KPI 카드
    park_mean_search = merged[selected_park].mean()
    park_total = merged["총이용객"].sum()
    park_corr = merged[selected_park].corr(merged["총이용객"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>총 이용객 (전체 기간)</h3>'
            f'<div class="value">{park_total:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>평균 월 이용객</h3>'
            f'<div class="value">{merged["총이용객"].mean():,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>{selected_park} 평균 검색량</h3>'
            f'<div class="value">{park_mean_search:.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>검색량↔이용객 상관</h3>'
            f'<div class="value">{park_corr:.3f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # 시계열 추이
    st.markdown('<h3 class="section-header">시계열 추이</h3>', unsafe_allow_html=True)

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(
        go.Scatter(
            x=merged["연월"], y=merged["총이용객"],
            name="총이용객", line=dict(color="#2E86AB", width=2.5),
        ),
        secondary_y=False,
    )
    fig_ts.add_trace(
        go.Scatter(
            x=merged["연월"], y=merged[selected_park],
            name=f"{selected_park} 검색량",
            line=dict(color="#E8505B", width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig_ts.update_layout(
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
    )
    fig_ts.update_yaxes(title_text="총이용객", secondary_y=False)
    fig_ts.update_yaxes(title_text="검색량", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    # 계절 & 월별
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<h3 class="section-header">계절별 분포</h3>', unsafe_allow_html=True)
        fig_season = px.box(
            merged, x="계절", y="총이용객",
            color="계절",
            color_discrete_sequence=["#26de81", "#fd9644", "#fc5c65", "#4b7bec"],
        )
        fig_season.update_layout(height=350, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_season, use_container_width=True)

    with col_b:
        st.markdown('<h3 class="section-header">월별 평균</h3>', unsafe_allow_html=True)
        monthly_avg = merged.groupby("월")["총이용객"].mean().reset_index()
        fig_monthly = px.bar(
            monthly_avg, x="월", y="총이용객",
            color="총이용객", color_continuous_scale="Blues",
        )
        fig_monthly.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # 공원별 검색량 비교
    st.markdown('<h3 class="section-header">공원별 평균 검색량 비교</h3>', unsafe_allow_html=True)
    park_means = merged[park_list].mean().sort_values(ascending=True).reset_index()
    park_means.columns = ["공원", "평균 검색량"]
    colors = ["#E8505B" if p == selected_park else "#2E86AB" for p in park_means["공원"]]
    fig_bar = go.Figure(
        go.Bar(x=park_means["평균 검색량"], y=park_means["공원"], orientation="h", marker_color=colors)
    )
    fig_bar.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    # 상관 히트맵
    st.markdown('<h3 class="section-header">공원 간 검색량 상관관계</h3>', unsafe_allow_html=True)
    corr_matrix = merged[park_list].corr()
    fig_heat = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig_heat.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================
# 🔬 t-test & VIF
# =============================================================
elif page == "🔬 t-test & VIF":

    st.markdown('<h3 class="section-header">t-test 기반 피처 선택</h3>', unsafe_allow_html=True)
    st.info(f"**{selected_park}** 기준: 총이용객 중앙값으로 High/Low 그룹 분리 → 각 피처 t-test")

    features = [c for c in feature_cols] + [selected_park]

    median_v = merged["총이용객"].median()
    high = merged[merged["총이용객"] >= median_v]
    low = merged[merged["총이용객"] < median_v]

    ttest_rows = []
    for col in features:
        x1 = pd.to_numeric(high[col], errors="coerce").dropna()
        x2 = pd.to_numeric(low[col], errors="coerce").dropna()
        t, p = stats.ttest_ind(x1, x2, nan_policy="omit")
        ttest_rows.append({"피처": col, "t_stat": round(t, 4), "p_value": round(p, 4)})

    ttest_df = pd.DataFrame(ttest_rows).sort_values("p_value")
    ttest_df["유의"] = ttest_df["p_value"].apply(lambda p: "✅ 유의" if p < 0.05 else "❌")

    sig_cols = ttest_df[ttest_df["p_value"] < 0.05]["피처"].tolist()
    no_vif_cols = sig_cols + ([selected_park] if selected_park not in sig_cols else [])

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_ttest = px.bar(
            ttest_df.head(20), x="p_value", y="피처", orientation="h",
            color=ttest_df.head(20)["p_value"].apply(lambda p: "유의 (p<0.05)" if p < 0.05 else "비유의"),
            color_discrete_map={"유의 (p<0.05)": "#E8505B", "비유의": "#4b7bec"},
        )
        fig_ttest.add_vline(x=0.05, line_dash="dash", line_color="black")
        fig_ttest.update_layout(height=500, template="plotly_white", showlegend=True)
        st.plotly_chart(fig_ttest, use_container_width=True)
    with col2:
        st.markdown(f"**유의한 피처: {len(sig_cols)}개**")
        st.dataframe(ttest_df, height=500, use_container_width=True)

    # VIF
    st.markdown("---")
    st.markdown('<h3 class="section-header">Stepwise VIF 제거 (편향 최소화)</h3>', unsafe_allow_html=True)
    st.info("VIF > 10인 변수 중 가장 큰 것만 **하나씩** 제거 → 정보 손실 최소화")

    if len(no_vif_cols) >= 2:
        X_sub = merged[no_vif_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        survived, removed = stepwise_vif(X_sub, threshold=10)

        if selected_park not in survived:
            survived.append(selected_park)

        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**VIF 통과 변수: {len(survived)}개**")
            st.write(survived)
        with c2:
            if removed:
                st.warning(f"**제거된 변수: {len(removed)}개**")
                for feat, vif_val in removed:
                    st.write(f"  - {feat} (VIF={vif_val})")
            else:
                st.success("제거된 변수 없음 (모든 VIF ≤ 10)")
    else:
        survived = no_vif_cols
        st.warning("변수가 2개 미만이라 VIF 적용 불가")

    # session state에 저장
    st.session_state["no_vif_cols"] = no_vif_cols
    st.session_state["vif_cols"] = survived


# =============================================================
# 🤖 모델 학습 & 비교
# =============================================================
elif page == "🤖 모델 학습 & 비교":

    st.markdown('<h3 class="section-header">모델 학습 & 성능 비교</h3>', unsafe_allow_html=True)

    with st.spinner("모델 학습 중... (5개 모델 × 2 데이터셋)"):

        # 데이터 준비
        X_orig = preprocess(merged[feature_cols].copy())
        y = merged["총이용객"]

        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_orig, y, test_size=0.2, random_state=42)
        sc_o = StandardScaler()
        Xo_tr_sc = sc_o.fit_transform(Xo_tr)
        Xo_te_sc = sc_o.transform(Xo_te)

        models = get_models()
        results = []
        trained_models = {}

        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(Xo_tr_sc, yo_tr)
            pred = m.predict(Xo_te_sc)
            rmse = np.sqrt(mean_squared_error(yo_te, pred))
            mae = mean_absolute_error(yo_te, pred)
            r2 = r2_score(yo_te, pred)
            results.append(["Original", name, round(rmse, 2), round(mae, 2), round(r2, 4)])
            trained_models[f"Original_{name}"] = m

        # Feature Importance Top 10 재학습
        gb_model = trained_models["Original_GradientBoosting"]
        fi_df = pd.DataFrame({
            "Feature": X_orig.columns,
            "Importance": gb_model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        top10 = fi_df.head(10)["Feature"].tolist()
        X_top = preprocess(merged[top10].copy())

        Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X_top, y, test_size=0.2, random_state=42)
        sc_t = StandardScaler()
        Xt_tr_sc = sc_t.fit_transform(Xt_tr)
        Xt_te_sc = sc_t.transform(Xt_te)

        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(Xt_tr_sc, yt_tr)
            pred = m.predict(Xt_te_sc)
            rmse = np.sqrt(mean_squared_error(yt_te, pred))
            mae = mean_absolute_error(yt_te, pred)
            r2 = r2_score(yt_te, pred)
            results.append(["FI_Top10", name, round(rmse, 2), round(mae, 2), round(r2, 4)])
            trained_models[f"FI_Top10_{name}"] = m

    res_df = pd.DataFrame(results, columns=["Dataset", "Model", "RMSE", "MAE", "R2"])
    res_df = res_df.sort_values("R2", ascending=False)

    # 성능 차트
    fig_perf = px.bar(
        res_df, x="Model", y="R2", color="Dataset",
        barmode="group",
        color_discrete_map={"Original": "#2E86AB", "FI_Top10": "#E8505B"},
        text="R2",
    )
    fig_perf.update_traces(textposition="outside", texttemplate="%{text:.3f}")
    fig_perf.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.dataframe(res_df, use_container_width=True)

    # Feature Importance
    st.markdown("---")
    st.markdown('<h3 class="section-header">Feature Importance (GradientBoosting)</h3>', unsafe_allow_html=True)

    fig_fi = px.bar(
        fi_df.head(15), x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig_fi.update_layout(height=450, template="plotly_white", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"**재학습에 사용된 Top 10 피처:** {top10}")

    # 세션 저장
    st.session_state["trained_models"] = trained_models
    st.session_state["Xo_tr_sc"] = Xo_tr_sc
    st.session_state["Xo_te_sc"] = Xo_te_sc
    st.session_state["yo_te"] = yo_te
    st.session_state["yo_tr"] = yo_tr
    st.session_state["X_orig"] = X_orig
    st.session_state["sc_o"] = sc_o
    st.session_state["fi_df"] = fi_df
    st.session_state["feature_cols"] = feature_cols
    st.session_state["Xo_te"] = Xo_te


# =============================================================
# 📈 잔차 도표 & 진단
# =============================================================
elif page == "📈 잔차 도표 & 진단":

    if "trained_models" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 학습 & 비교** 탭을 실행해주세요.")
        st.stop()

    trained_models = st.session_state["trained_models"]
    Xo_te_sc = st.session_state["Xo_te_sc"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    yo_te = st.session_state["yo_te"]
    yo_tr = st.session_state["yo_tr"]
    X_orig = st.session_state["X_orig"]
    feature_cols_s = st.session_state["feature_cols"]

    model_choice = st.selectbox(
        "잔차 분석할 모델 선택",
        [k.replace("Original_", "") for k in trained_models if k.startswith("Original_")],
    )
    model = trained_models[f"Original_{model_choice}"]
    y_pred = model.predict(Xo_te_sc)
    residuals = yo_te.values - y_pred

    # 잔차 도표 4종
    st.markdown('<h3 class="section-header">잔차 도표 4종 (Minitab 스타일)</h3>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", s=40, color="#2E86AB")
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("① Residuals vs Fitted", fontweight="bold")

    probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("② Normal Q-Q Plot", fontweight="bold")
    axes[0, 1].get_lines()[1].set_color("red")

    axes[1, 0].hist(residuals, bins=12, edgecolor="black", alpha=0.7, color="#2E86AB")
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("③ Histogram of Residuals", fontweight="bold")

    axes[1, 1].plot(range(len(residuals)), residuals, "o-", markersize=4, alpha=0.7, color="#2E86AB")
    axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 1].set_xlabel("Observation Order")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("④ Residuals vs Order", fontweight="bold")

    plt.suptitle(f"Residual Plots ({model_choice})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 정규성 검정
    stat_sw, p_sw = stats.shapiro(residuals)
    if p_sw > 0.05:
        st.success(f"Shapiro-Wilk: p={p_sw:.4f} → 잔차가 정규분포를 따름")
    else:
        st.warning(f"Shapiro-Wilk: p={p_sw:.4f} → 잔차가 정규분포에서 벗어남 (소표본에서 흔함)")

    # Learning Curve
    st.markdown("---")
    st.markdown('<h3 class="section-header">Learning Curve (과적합 진단)</h3>', unsafe_allow_html=True)

    lc_models = {
        "Ridge": Ridge(alpha=10),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, m) in enumerate(lc_models.items()):
        ax = axes_lc[idx]
        train_sizes, train_sc, val_sc = learning_curve(
            m, Xo_tr_sc, yo_tr,
            train_sizes=np.linspace(0.2, 1.0, 8),
            cv=5, scoring="r2", random_state=42, n_jobs=-1,
        )
        ax.plot(train_sizes, train_sc.mean(axis=1), "o-", label="Train R²", color="#2E86AB")
        ax.fill_between(train_sizes,
                        train_sc.mean(axis=1) - train_sc.std(axis=1),
                        train_sc.mean(axis=1) + train_sc.std(axis=1),
                        alpha=0.15, color="#2E86AB")
        ax.plot(train_sizes, val_sc.mean(axis=1), "s-", label="Val R²", color="#E8505B")
        ax.fill_between(train_sizes,
                        val_sc.mean(axis=1) - val_sc.std(axis=1),
                        val_sc.mean(axis=1) + val_sc.std(axis=1),
                        alpha=0.15, color="#E8505B")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Training Size")
        ax.set_ylabel("R²")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.5, 1.1)

    plt.suptitle("Learning Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_lc)
    plt.close()

    st.markdown("""
    **해석 가이드:**
    - Train↑ Val↑ 가까움 → 좋은 모델
    - Train↑ Val↓ → **과적합** (데이터 추가 필요)
    - 둘 다↓ → **과소적합** (모델 복잡도 부족)
    """)


# =============================================================
# 💡 SHAP 해석
# =============================================================
elif page == "💡 SHAP 해석":

    if "trained_models" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 학습 & 비교** 탭을 실행해주세요.")
        st.stop()

    st.markdown('<h3 class="section-header">SHAP Summary Plot</h3>', unsafe_allow_html=True)
    st.info("각 피처가 개별 예측에 얼마나 기여했는지 시각화합니다. 빨간점=피처값 높음, 파란점=낮음")

    gb = st.session_state["trained_models"]["Original_GradientBoosting"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    Xo_te_sc = st.session_state["Xo_te_sc"]
    Xo_te = st.session_state["Xo_te"]
    X_orig = st.session_state["X_orig"]

    with st.spinner("SHAP 값 계산 중..."):
        explainer = shap.Explainer(gb, Xo_tr_sc)
        shap_values = explainer(Xo_te_sc)

    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, Xo_te,
        feature_names=X_orig.columns.tolist(),
        show=False,
    )
    st.pyplot(fig_shap)
    plt.close()

    # Bar plot
    st.markdown("---")
    st.markdown('<h3 class="section-header">SHAP 평균 기여도 (Bar)</h3>', unsafe_allow_html=True)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig_bar)
    plt.close()


# =============================================================
# 🎯 Conformal Prediction
# =============================================================
elif page == "🎯 Conformal Prediction":

    st.markdown('<h3 class="section-header">Conformal Prediction (90% 예측 구간)</h3>', unsafe_allow_html=True)

    st.markdown("""
    **원리:** Train → Calibration set 잔차 분포 학습 → Test에 예측 구간 부여

    - α = 0.1 → 목표 커버리지 90%
    - 예측값 ± q_hat (잔차 90번째 백분위수)
    """)

    alpha = st.slider("α (1 - 커버리지)", 0.05, 0.30, 0.10, 0.05)

    X_cp = preprocess(merged[feature_cols].copy())
    y_cp = merged["총이용객"]

    X_tr, X_temp, y_tr, y_temp = train_test_split(X_cp, y_cp, test_size=0.4, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_cal_sc = sc.transform(X_cal)
    X_te_sc = sc.transform(X_te)

    cp_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    cp_model.fit(X_tr_sc, y_tr)

    cal_pred = cp_model.predict(X_cal_sc)
    cal_resid = np.abs(y_cal.values - cal_pred)
    q_hat = np.quantile(cal_resid, 1 - alpha)

    te_pred = cp_model.predict(X_te_sc)
    lower = te_pred - q_hat
    upper = te_pred + q_hat
    coverage = np.mean((y_te.values >= lower) & (y_te.values <= upper))

    c1, c2, c3 = st.columns(3)
    c1.metric("q_hat (구간 반폭)", f"{q_hat:,.0f}")
    c2.metric("목표 커버리지", f"{(1-alpha)*100:.0f}%")
    c3.metric("실제 커버리지", f"{coverage*100:.1f}%")

    idx = np.arange(len(y_te))
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(x=idx, y=y_te.values, mode="lines+markers", name="Actual", line=dict(color="#2E86AB")))
    fig_cp.add_trace(go.Scatter(x=idx, y=te_pred, mode="lines+markers", name="Predicted", line=dict(color="#E8505B")))
    fig_cp.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int((1-alpha)*100)}% Interval",
    ))
    fig_cp.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_cp, use_container_width=True)


# =============================================================
# 🧠 LSTM
# =============================================================
elif page == "🧠 LSTM (딥러닝)":

    st.markdown('<h3 class="section-header">LSTM (시계열 딥러닝)</h3>', unsafe_allow_html=True)

    st.warning(f"⚠️ 데이터가 {len(merged)}개월로 매우 적습니다. LSTM 결과는 **참고용**입니다.")

    st.markdown("""
    **LSTM (Long Short-Term Memory)**
    - RNN의 한 종류로 시계열에서 이전 시점 정보를 기억하며 예측
    - `look_back=3` → 최근 3개월 데이터로 다음 달 예측
    """)

    look_back = st.slider("look_back (과거 몇 개월?)", 2, 6, 3)

    if st.button("🚀 LSTM 학습 시작", type="primary"):

        with st.spinner("LSTM 학습 중... (약 30초)"):
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            ts = merged.sort_values("연월")["총이용객"].values.reshape(-1, 1)
            sc_lstm = MinMaxScaler(feature_range=(0, 1))
            ts_sc = sc_lstm.fit_transform(ts)

            X_seq, y_seq = [], []
            for i in range(look_back, len(ts_sc)):
                X_seq.append(ts_sc[i - look_back:i, 0])
                y_seq.append(ts_sc[i, 0])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

            split = int(len(X_seq) * 0.8)
            Xtr, Xte = X_seq[:split], X_seq[split:]
            ytr, yte = y_seq[:split], y_seq[split:]

            model_lstm = Sequential([
                LSTM(50, activation="relu", input_shape=(look_back, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(30, activation="relu"),
                Dropout(0.2),
                Dense(1),
            ])
            model_lstm.compile(optimizer="adam", loss="mse")

            history = model_lstm.fit(
                Xtr, ytr, epochs=100, batch_size=4,
                validation_split=0.2, verbose=0,
            )

        y_pred_lstm = model_lstm.predict(Xte, verbose=0).flatten()
        y_real = sc_lstm.inverse_transform(yte.reshape(-1, 1)).flatten()
        y_pred_real = sc_lstm.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

        rmse_l = np.sqrt(mean_squared_error(y_real, y_pred_real))
        mae_l = mean_absolute_error(y_real, y_pred_real)
        r2_l = r2_score(y_real, y_pred_real)

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse_l:,.0f}")
        c2.metric("MAE", f"{mae_l:,.0f}")
        c3.metric("R²", f"{r2_l:.4f}")

        if r2_l < 0:
            st.error("R²가 음수 → LSTM이 평균보다 못한 예측. 데이터 부족이 원인입니다.")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss", line=dict(color="#2E86AB")))
            fig_loss.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss", line=dict(color="#E8505B")))
            fig_loss.update_layout(title="학습 Loss 곡선", height=350, template="plotly_white")
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_b:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(y=y_real, name="Actual", mode="lines+markers", line=dict(color="#2E86AB")))
            fig_pred.add_trace(go.Scatter(y=y_pred_real, name="LSTM Predicted", mode="lines+markers", line=dict(color="#E8505B")))
            fig_pred.update_layout(title="LSTM: 실제 vs 예측", height=350, template="plotly_white")
            st.plotly_chart(fig_pred, use_container_width=True)

    else:
        st.markdown("👆 버튼을 눌러 LSTM 학습을 시작하세요.")
