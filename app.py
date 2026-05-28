"""
한강공원 이용객 분석 대시보드
============================
공원별 EDA → t-test → VIF → 모델 학습 & 비교 → 잔차 도표 → SHAP
→ Conformal Prediction → Bootstrap 95% CI → Nested CV
(LSTM 제거 / pkl 단일 모델 사용)
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
import glob
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

from scipy import stats
from scipy.stats import probplot
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    KFold,
    cross_val_score,
)
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────
import platform

sys_name = platform.system()
if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
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
    .main-header h1 { font-size: 2.2rem; font-weight: 900; margin: 0; letter-spacing: -1px; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }

    .metric-card {
        background: white;
        border: 1px solid #e8ecef;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card h3    { font-size: 0.85rem; color: #666; margin: 0 0 0.3rem 0; font-weight: 400; }
    .metric-card .value{ font-size: 1.6rem; font-weight: 700; color: #1A1A2E; }

    .section-header {
        border-left: 4px solid #2E86AB;
        padding-left: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 700;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F2027 0%, #203A43 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# pkl 모델 로드
# ─────────────────────────────────────────────
@st.cache_resource
def load_pkl_model():
    """model/ 폴더에서 첫 번째 pkl 파일을 자동으로 로드"""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None, None
    pkl_path = pkl_files[0]
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj, os.path.basename(pkl_path)


pkl_model, pkl_name = load_pkl_model()


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    df = pd.read_csv(os.path.join(data_dir, "users.csv"), encoding="utf-8")
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

    trend = pd.read_excel(os.path.join(data_dir, "trend.xlsx"))
    trend.rename(columns={"날짜": "연월"}, inplace=True)
    trend["연월"] = pd.to_datetime(trend["연월"])

    merged = pd.merge(monthly, trend, on="연월", how="inner")
    merged["검색량"] = merged["한강공원"]
    merged["월"] = merged["연월"].dt.month

    def get_season(m):
        if m in [3, 4, 5]:   return "봄"
        elif m in [6, 7, 8]: return "여름"
        elif m in [9,10,11]: return "가을"
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
# 유틸 함수
# ─────────────────────────────────────────────
def preprocess(X):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X


def stepwise_vif(X_df, threshold=10):
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


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """Bootstrap으로 지표의 신뢰구간을 계산"""
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            continue
        scores.append(s)
    scores = np.array(scores)
    alpha = (100 - ci) / 2
    lo = np.percentile(scores, alpha)
    hi = np.percentile(scores, 100 - alpha)
    return np.mean(scores), lo, hi


# ─────────────────────────────────────────────
# 지도 & 공원 선택
# ─────────────────────────────────────────────
st.markdown("## 🗺️ 한강공원 선택")

parks = {
    "강서한강공원":   [37.588, 126.815],
    "양화한강공원":   [37.543, 126.901],
    "난지한강공원":   [37.568, 126.876],
    "망원한강공원":   [37.555, 126.897],
    "여의도한강공원": [37.528, 126.932],
    "이촌한강공원":   [37.517, 126.973],
    "반포한강공원":   [37.510, 126.995],
    "잠원한강공원":   [37.519, 127.011],
    "잠실한강공원":   [37.520, 127.086],
    "뚝섬한강공원":   [37.529, 127.072],
    "광나루한강공원": [37.548, 127.118],
}

m_map = folium.Map(location=[37.53, 126.98], zoom_start=11, tiles="CartoDB positron")
for park, coord in parks.items():
    folium.Marker(
        location=coord, tooltip=park, popup=park,
        icon=folium.Icon(color="blue", icon="tree-deciduous"),
    ).add_to(m_map)

map_data = st_folium(m_map, width=1000, height=500)
selected_park = "여의도한강공원"
if map_data["last_object_clicked_popup"]:
    selected_park = map_data["last_object_clicked_popup"]
st.success(f"선택된 공원: {selected_park}")

# pkl 모델 상태 표시
if pkl_model is not None:
    st.info(f"🤖 로드된 모델: **{pkl_name}**")
else:
    st.warning("⚠️ model/ 폴더에 pkl 파일이 없습니다.")

page = st.radio(
    "분석 메뉴",
    [
        "📊 EDA (탐색적 분석)",
        "🔬 t-test & VIF",
        "🤖 모델 학습 & 비교",
        "📈 잔차 도표 & 진단",
        "💡 SHAP 해석",
        "🎯 Conformal Prediction",
        "📐 Bootstrap 95% CI",
        "🔁 Nested CV",
    ],
    horizontal=True,
)

st.markdown(
    f"""
<div class="main-header">
    <h1>🏞️ 한강공원 이용객 분석 대시보드</h1>
    <p>현재 선택: <strong>{selected_park}</strong> · 11개 한강공원 비교 분석 · ML 예측</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 공통 변수
# ─────────────────────────────────────────────
leakage = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
feature_cols = [c for c in num_cols if c not in leakage]


# =============================================================
# 📊 EDA
# =============================================================
if page == "📊 EDA (탐색적 분석)":

    park_mean_search = merged[selected_park].mean()
    park_total = merged["총이용객"].sum()
    park_corr = merged[selected_park].corr(merged["총이용객"])

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "총 이용객 (전체 기간)",        f"{park_total:,.0f}"),
        (c2, "평균 월 이용객",               f"{merged['총이용객'].mean():,.0f}"),
        (c3, f"{selected_park} 평균 검색량", f"{park_mean_search:.1f}"),
        (c4, "검색량↔이용객 상관",           f"{park_corr:.3f}"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{label}</h3>'
                f'<div class="value">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('<h3 class="section-header">시계열 추이</h3>', unsafe_allow_html=True)

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(
        go.Scatter(x=merged["연월"], y=merged["총이용객"], name="총이용객",
                   line=dict(color="#2E86AB", width=2.5)),
        secondary_y=False,
    )
    fig_ts.add_trace(
        go.Scatter(x=merged["연월"], y=merged[selected_park],
                   name=f"{selected_park} 검색량",
                   line=dict(color="#E8505B", width=2, dash="dot")),
        secondary_y=True,
    )
    fig_ts.update_layout(height=400, template="plotly_white", legend=dict(orientation="h", y=1.12))
    fig_ts.update_yaxes(title_text="총이용객", secondary_y=False)
    fig_ts.update_yaxes(title_text="검색량", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<h3 class="section-header">계절별 분포</h3>', unsafe_allow_html=True)
        fig_season = px.box(
            merged, x="계절", y="총이용객", color="계절",
            color_discrete_sequence=["#26de81", "#fd9644", "#fc5c65", "#4b7bec"],
        )
        fig_season.update_layout(height=350, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_season, use_container_width=True)

    with col_b:
        st.markdown('<h3 class="section-header">월별 평균</h3>', unsafe_allow_html=True)
        monthly_avg = merged.groupby("월")["총이용객"].mean().reset_index()
        fig_monthly = px.bar(monthly_avg, x="월", y="총이용객",
                             color="총이용객", color_continuous_scale="Blues")
        fig_monthly.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown('<h3 class="section-header">공원별 평균 검색량 비교</h3>', unsafe_allow_html=True)
    park_means = merged[park_list].mean().sort_values(ascending=True).reset_index()
    park_means.columns = ["공원", "평균 검색량"]
    colors = ["#E8505B" if p == selected_park else "#2E86AB" for p in park_means["공원"]]
    fig_bar = go.Figure(
        go.Bar(x=park_means["평균 검색량"], y=park_means["공원"], orientation="h", marker_color=colors)
    )
    fig_bar.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<h3 class="section-header">공원 간 검색량 상관관계</h3>', unsafe_allow_html=True)
    corr_matrix = merged[park_list].corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
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
    low  = merged[merged["총이용객"] < median_v]

    ttest_rows = []
    for col in features:
        x1 = pd.to_numeric(high[col], errors="coerce").dropna()
        x2 = pd.to_numeric(low[col],  errors="coerce").dropna()
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

    st.session_state["no_vif_cols"] = no_vif_cols
    st.session_state["vif_cols"]    = survived


# =============================================================
# 🤖 모델 학습 & 비교  (pkl 모델 단독 사용)
# =============================================================
elif page == "🤖 모델 학습 & 비교":

    st.markdown('<h3 class="section-header">모델 학습 & 성능 비교</h3>', unsafe_allow_html=True)

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다. 파일을 추가해주세요.")
        st.stop()

    st.info(f"📦 사용 모델: **{pkl_name}**")

    with st.spinner("모델 예측 중..."):
        X_orig = preprocess(merged[feature_cols].copy())
        y = merged["총이용객"]

        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_orig, y, test_size=0.2, random_state=42)
        sc_o = StandardScaler()
        Xo_tr_sc = sc_o.fit_transform(Xo_tr)
        Xo_te_sc = sc_o.transform(Xo_te)

        # pkl 모델이 Pipeline이면 scaler 포함, 아니면 스케일된 데이터 사용
        try:
            pkl_model.fit(Xo_tr_sc, yo_tr)
            pred_pkl = pkl_model.predict(Xo_te_sc)
        except Exception:
            pkl_model.fit(Xo_tr, yo_tr)
            pred_pkl = pkl_model.predict(Xo_te)

        rmse = np.sqrt(mean_squared_error(yo_te, pred_pkl))
        mae  = mean_absolute_error(yo_te, pred_pkl)
        r2   = r2_score(yo_te, pred_pkl)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.0f}")
    c2.metric("MAE",  f"{mae:,.0f}")
    c3.metric("R²",   f"{r2:.4f}")

    # 실제 vs 예측 차트
    st.markdown('<h3 class="section-header">실제 vs 예측</h3>', unsafe_allow_html=True)
    idx = np.arange(len(yo_te))
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=idx, y=yo_te.values,  mode="lines+markers", name="실제",
                                  line=dict(color="#2E86AB", width=2)))
    fig_pred.add_trace(go.Scatter(x=idx, y=pred_pkl, mode="lines+markers", name="예측 (pkl)",
                                  line=dict(color="#E8505B", width=2, dash="dot")))
    fig_pred.update_layout(height=400, template="plotly_white",
                           xaxis_title="샘플 인덱스", yaxis_title="총이용객")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Feature Importance (가능한 경우)
    try:
        fi = None
        if hasattr(pkl_model, "feature_importances_"):
            fi = pkl_model.feature_importances_
        elif hasattr(pkl_model, "named_steps"):
            est = list(pkl_model.named_steps.values())[-1]
            if hasattr(est, "feature_importances_"):
                fi = est.feature_importances_
            elif hasattr(est, "coef_"):
                fi = np.abs(est.coef_)
        elif hasattr(pkl_model, "coef_"):
            fi = np.abs(pkl_model.coef_)

        if fi is not None:
            st.markdown('<h3 class="section-header">Feature Importance</h3>', unsafe_allow_html=True)
            fi_df = pd.DataFrame({"Feature": X_orig.columns, "Importance": fi})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Viridis")
            fig_fi.update_layout(height=450, template="plotly_white", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        pass

    # 세션 저장
    st.session_state.update({
        "pkl_model":    pkl_model,
        "Xo_tr_sc":     Xo_tr_sc,
        "Xo_te_sc":     Xo_te_sc,
        "yo_te":        yo_te,
        "yo_tr":        yo_tr,
        "X_orig":       X_orig,
        "sc_o":         sc_o,
        "feature_cols": feature_cols,
        "Xo_te":        Xo_te,
        "pred_pkl":     pred_pkl,
    })


# =============================================================
# 📈 잔차 도표 & 진단
# =============================================================
elif page == "📈 잔차 도표 & 진단":

    if "pkl_model" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 학습 & 비교** 탭을 실행해주세요.")
        st.stop()

    Xo_te_sc = st.session_state["Xo_te_sc"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    yo_te    = st.session_state["yo_te"]
    yo_tr    = st.session_state["yo_tr"]
    X_orig   = st.session_state["X_orig"]
    pred_pkl = st.session_state["pred_pkl"]
    residuals = yo_te.values - pred_pkl

    st.markdown(f'<h3 class="section-header">잔차 도표 4종 · {pkl_name}</h3>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(pred_pkl, residuals, alpha=0.6, edgecolors="k", s=40, color="#2E86AB")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_xlabel("Fitted Values"); axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("① Residuals vs Fitted", fontweight="bold")

    probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("② Normal Q-Q Plot", fontweight="bold")
    axes[0, 1].get_lines()[1].set_color("red")

    axes[1, 0].hist(residuals, bins=12, edgecolor="black", alpha=0.7, color="#2E86AB")
    axes[1, 0].set_xlabel("Residuals"); axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("③ Histogram of Residuals", fontweight="bold")

    axes[1, 1].plot(range(len(residuals)), residuals, "o-", markersize=4, alpha=0.7, color="#2E86AB")
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 1].set_xlabel("Observation Order"); axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("④ Residuals vs Order", fontweight="bold")

    plt.suptitle(f"Residual Plots ({pkl_name})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    stat_sw, p_sw = stats.shapiro(residuals)
    if p_sw > 0.05:
        st.success(f"Shapiro-Wilk: p={p_sw:.4f} → 잔차가 정규분포를 따름")
    else:
        st.warning(f"Shapiro-Wilk: p={p_sw:.4f} → 잔차가 정규분포에서 벗어남 (소표본에서 흔함)")


# =============================================================
# 💡 SHAP 해석
# =============================================================
elif page == "💡 SHAP 해석":

    if "pkl_model" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 학습 & 비교** 탭을 실행해주세요.")
        st.stop()

    st.markdown('<h3 class="section-header">SHAP Summary Plot</h3>', unsafe_allow_html=True)
    st.info("각 피처가 개별 예측에 얼마나 기여했는지 시각화합니다. 빨간점=피처값 높음, 파란점=낮음")

    model_for_shap = st.session_state["pkl_model"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    Xo_te_sc = st.session_state["Xo_te_sc"]
    Xo_te    = st.session_state["Xo_te"]
    X_orig   = st.session_state["X_orig"]

    # Pipeline인 경우 마지막 estimator 추출
    if hasattr(model_for_shap, "named_steps"):
        est_shap = list(model_for_shap.named_steps.values())[-1]
    else:
        est_shap = model_for_shap

    with st.spinner("SHAP 값 계산 중..."):
        try:
            explainer   = shap.Explainer(est_shap, Xo_tr_sc)
            shap_values = explainer(Xo_te_sc)
        except Exception:
            explainer   = shap.KernelExplainer(est_shap.predict, Xo_tr_sc[:50])
            shap_values = explainer.shap_values(Xo_te_sc)

    fig_shap, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, Xo_te,
                      feature_names=X_orig.columns.tolist(), show=False)
    st.pyplot(fig_shap)
    plt.close()

    st.markdown("---")
    st.markdown('<h3 class="section-header">SHAP 평균 기여도 (Bar)</h3>', unsafe_allow_html=True)
    fig_bar2, _ = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig_bar2)
    plt.close()


# =============================================================
# 🎯 Conformal Prediction
# =============================================================
elif page == "🎯 Conformal Prediction":

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다.")
        st.stop()

    st.markdown('<h3 class="section-header">Conformal Prediction (90% 예측 구간)</h3>', unsafe_allow_html=True)
    st.markdown("""
**원리:** Train → Calibration set 잔차 분포 학습 → Test에 예측 구간 부여

- α = 0.1 → 목표 커버리지 90%
- 예측값 ± q_hat (잔차 90번째 백분위수)
""")

    alpha_cp = st.slider("α (1 - 커버리지)", 0.05, 0.30, 0.10, 0.05)

    X_cp = preprocess(merged[feature_cols].copy())
    y_cp = merged["총이용객"]

    X_tr, X_temp, y_tr, y_temp = train_test_split(X_cp, y_cp, test_size=0.4, random_state=42)
    X_cal, X_te, y_cal, y_te   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    sc = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr)
    X_cal_sc = sc.transform(X_cal)
    X_te_sc  = sc.transform(X_te)

    cp_model = copy.deepcopy(pkl_model)
    try:
        cp_model.fit(X_tr_sc, y_tr)
        cal_pred = cp_model.predict(X_cal_sc)
        te_pred  = cp_model.predict(X_te_sc)
    except Exception:
        cp_model.fit(X_tr, y_tr)
        cal_pred = cp_model.predict(X_cal)
        te_pred  = cp_model.predict(X_te)

    cal_resid = np.abs(y_cal.values - cal_pred)
    q_hat     = np.quantile(cal_resid, 1 - alpha_cp)
    lower     = te_pred - q_hat
    upper     = te_pred + q_hat
    coverage  = np.mean((y_te.values >= lower) & (y_te.values <= upper))

    c1, c2, c3 = st.columns(3)
    c1.metric("q_hat (구간 반폭)", f"{q_hat:,.0f}")
    c2.metric("목표 커버리지",     f"{(1-alpha_cp)*100:.0f}%")
    c3.metric("실제 커버리지",     f"{coverage*100:.1f}%")

    idx = np.arange(len(y_te))
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(x=idx, y=y_te.values, mode="lines+markers",
                                name="Actual", line=dict(color="#2E86AB")))
    fig_cp.add_trace(go.Scatter(x=idx, y=te_pred, mode="lines+markers",
                                name="Predicted", line=dict(color="#E8505B")))
    fig_cp.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int((1-alpha_cp)*100)}% Interval",
    ))
    fig_cp.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_cp, use_container_width=True)


# =============================================================
# 📐 Bootstrap 95% CI
# =============================================================
elif page == "📐 Bootstrap 95% CI":

    st.markdown('<h3 class="section-header">Bootstrap 95% 신뢰구간 (성능 지표)</h3>', unsafe_allow_html=True)
    st.markdown("""
**원리:** 테스트셋에서 복원 추출(n=1000회) → 매 회 RMSE/MAE/R² 계산 → 2.5th~97.5th 백분위수를 CI로 사용

- 단일 분할의 운에 의존하지 않고 **지표의 안정성(변동폭)** 을 정량화
- CI 폭이 좁을수록 모델 성능이 일관적임을 의미
""")

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다.")
        st.stop()

    n_boot = st.slider("Bootstrap 반복 횟수", 500, 3000, 1000, 500)

    X_bs = preprocess(merged[feature_cols].copy())
    y_bs = merged["총이용객"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_bs, y_bs, test_size=0.2, random_state=42)

    sc_bs = StandardScaler()
    X_tr_sc = sc_bs.fit_transform(X_tr)
    X_te_sc = sc_bs.transform(X_te)

    bs_model = copy.deepcopy(pkl_model)
    try:
        bs_model.fit(X_tr_sc, y_tr)
        y_pred_bs = bs_model.predict(X_te_sc)
    except Exception:
        bs_model.fit(X_tr, y_tr)
        y_pred_bs = bs_model.predict(X_te)

    with st.spinner(f"Bootstrap {n_boot}회 반복 중..."):
        rmse_fn = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        mae_fn  = lambda yt, yp: mean_absolute_error(yt, yp)
        r2_fn   = lambda yt, yp: r2_score(yt, yp)

        y_true_arr = y_te.values

        rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(y_true_arr, y_pred_bs, rmse_fn, n_boot)
        mae_mean,  mae_lo,  mae_hi  = bootstrap_ci(y_true_arr, y_pred_bs, mae_fn,  n_boot)
        r2_mean,   r2_lo,   r2_hi   = bootstrap_ci(y_true_arr, y_pred_bs, r2_fn,   n_boot)

        # 분포 수집 (시각화용)
        rng = np.random.RandomState(42)
        n = len(y_true_arr)
        rmse_dist, mae_dist, r2_dist = [], [], []
        for _ in range(n_boot):
            idx_b = rng.randint(0, n, n)
            try:
                rmse_dist.append(rmse_fn(y_true_arr[idx_b], y_pred_bs[idx_b]))
                mae_dist.append(mae_fn(y_true_arr[idx_b],   y_pred_bs[idx_b]))
                r2_dist.append(r2_fn(y_true_arr[idx_b],     y_pred_bs[idx_b]))
            except Exception:
                pass
        rmse_dist = np.array(rmse_dist)
        mae_dist  = np.array(mae_dist)
        r2_dist   = np.array(r2_dist)

    # ── KPI 카드
    st.markdown("#### 📊 지표별 평균 및 95% 신뢰구간")
    k1, k2, k3 = st.columns(3)
    k1.metric("RMSE (mean)",     f"{rmse_mean:,.0f}",
              f"CI: [{rmse_lo:,.0f} ~ {rmse_hi:,.0f}]")
    k2.metric("MAE (mean)",      f"{mae_mean:,.0f}",
              f"CI: [{mae_lo:,.0f} ~ {mae_hi:,.0f}]")
    k3.metric("R² (mean)",       f"{r2_mean:.4f}",
              f"CI: [{r2_lo:.4f} ~ {r2_hi:.4f}]")

    st.markdown("---")

    # ── 분포 히스토그램 + CI 음영
    st.markdown("#### 📈 Bootstrap 분포 (히스토그램 + 95% CI 음영)")

    fig_boot, axes_b = plt.subplots(1, 3, figsize=(15, 5))

    def plot_boot_hist(ax, dist, lo, hi, mean_v, label, color):
        ax.hist(dist, bins=40, color=color, alpha=0.65, edgecolor="none")
        ymax = ax.get_ylim()[1]
        ax.axvspan(lo, hi, alpha=0.25, color=color, label=f"95% CI")
        ax.axvline(mean_v, color="black",  linestyle="--", linewidth=1.5, label=f"Mean={mean_v:.3g}")
        ax.axvline(lo,     color="crimson",linestyle=":",  linewidth=1.2, label=f"Lo={lo:.3g}")
        ax.axvline(hi,     color="crimson",linestyle=":",  linewidth=1.2, label=f"Hi={hi:.3g}")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Bootstrap Distribution\n{label}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plot_boot_hist(axes_b[0], rmse_dist, rmse_lo, rmse_hi, rmse_mean, "RMSE", "#2E86AB")
    plot_boot_hist(axes_b[1], mae_dist,  mae_lo,  mae_hi,  mae_mean,  "MAE",  "#26de81")
    plot_boot_hist(axes_b[2], r2_dist,   r2_lo,   r2_hi,   r2_mean,   "R²",   "#E8505B")

    plt.suptitle(f"Bootstrap {n_boot}회 · {pkl_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_boot)
    plt.close()

    # ── CI 범위 요약 바 차트
    st.markdown("---")
    st.markdown("#### 📊 신뢰구간 폭(Width) 비교")

    ci_summary = pd.DataFrame({
        "지표":  ["RMSE", "MAE", "R²"],
        "Mean":  [rmse_mean, mae_mean, r2_mean],
        "CI_lo": [rmse_lo, mae_lo, r2_lo],
        "CI_hi": [rmse_hi, mae_hi, r2_hi],
        "Width": [rmse_hi - rmse_lo, mae_hi - mae_lo, r2_hi - r2_lo],
    })

    fig_ci = go.Figure()
    for _, row in ci_summary.iterrows():
        fig_ci.add_trace(go.Bar(
            name=row["지표"],
            x=[row["지표"]],
            y=[row["Width"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["CI_hi"] - row["Mean"]],
                arrayminus=[row["Mean"] - row["CI_lo"]],
            ),
            text=f"±{row['Width']:.3g}",
            textposition="outside",
        ))
    fig_ci.update_layout(
        height=380, template="plotly_white",
        title="95% CI 폭 (작을수록 안정적인 모델)",
        showlegend=False,
        yaxis_title="CI Width",
    )
    st.plotly_chart(fig_ci, use_container_width=True)

    st.dataframe(
        ci_summary.style.format({"Mean": "{:.4g}", "CI_lo": "{:.4g}",
                                  "CI_hi": "{:.4g}", "Width": "{:.4g}"}),
        use_container_width=True,
    )


# =============================================================
# 🔁 Nested CV
# =============================================================
elif page == "🔁 Nested CV":

    st.markdown('<h3 class="section-header">Nested Cross-Validation (신뢰도 향상)</h3>', unsafe_allow_html=True)
    st.markdown("""
**구조:**
- **Outer loop (K=5)**: 모델 성능의 불편 추정 → fold별 R² / RMSE
- **Inner loop (K=3)**: 하이퍼파라미터 탐색 공간을 pkl 모델 고정 파라미터로 대체 (단일 모델 평가)
- outer fold별 분포 차이가 작을수록 → **일반화 성능이 안정적**
""")

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다.")
        st.stop()

    outer_k = st.slider("Outer Fold 수 (K)", 3, 10, 5)
    inner_k = st.slider("Inner Fold 수 (K)", 2, 5, 3)

    if st.button("🚀 Nested CV 실행", type="primary"):

        X_ncv = preprocess(merged[feature_cols].copy())
        y_ncv = merged["총이용객"]

        sc_ncv = StandardScaler()
        X_sc   = sc_ncv.fit_transform(X_ncv)

        outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=42)

        outer_r2_list, outer_rmse_list = [], []
        inner_r2_all = []   # inner fold 평균 (per outer fold)
        fold_details = []

        with st.spinner(f"Nested CV 실행 중... (outer {outer_k} × inner {inner_k})"):
            for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X_sc), 1):
                X_tr_o, X_te_o = X_sc[tr_idx], X_sc[te_idx]
                y_tr_o, y_te_o = y_ncv.values[tr_idx], y_ncv.values[te_idx]

                # ── Inner CV: pkl 모델 그대로 평가 (고정 파라미터)
                inner_scores = []
                for tr_i, val_i in inner_cv.split(X_tr_o):
                    m_in = copy.deepcopy(pkl_model)
                    try:
                        m_in.fit(X_tr_o[tr_i], y_tr_o[tr_i])
                        p_in = m_in.predict(X_tr_o[val_i])
                    except Exception:
                        # 비스케일 fallback
                        X_orig_in = X_ncv.values
                        m_in.fit(X_orig_in[tr_idx][tr_i], y_tr_o[tr_i])
                        p_in = m_in.predict(X_orig_in[tr_idx][val_i])
                    inner_scores.append(r2_score(y_tr_o[val_i], p_in))
                inner_mean_r2 = np.mean(inner_scores)
                inner_r2_all.append(inner_mean_r2)

                # ── Outer: 전체 train으로 재학습 → test 평가
                m_out = copy.deepcopy(pkl_model)
                try:
                    m_out.fit(X_tr_o, y_tr_o)
                    p_out = m_out.predict(X_te_o)
                except Exception:
                    X_orig_all = X_ncv.values
                    m_out.fit(X_orig_all[tr_idx], y_tr_o)
                    p_out = m_out.predict(X_orig_all[te_idx])

                r2_out   = r2_score(y_te_o, p_out)
                rmse_out = np.sqrt(mean_squared_error(y_te_o, p_out))
                outer_r2_list.append(r2_out)
                outer_rmse_list.append(rmse_out)

                fold_details.append({
                    "Fold":       fold_i,
                    "Inner R² (평균)": round(inner_mean_r2, 4),
                    "Outer R²":        round(r2_out,        4),
                    "Outer RMSE":      round(rmse_out,      2),
                    "Gap (Inner-Outer)": round(inner_mean_r2 - r2_out, 4),
                })

        outer_r2   = np.array(outer_r2_list)
        outer_rmse = np.array(outer_rmse_list)

        # ── KPI 카드
        st.markdown("#### 📊 Nested CV 결과 요약")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Outer R² (mean ± std)",
                  f"{outer_r2.mean():.4f}",
                  f"± {outer_r2.std():.4f}")
        k2.metric("Outer RMSE (mean ± std)",
                  f"{outer_rmse.mean():,.0f}",
                  f"± {outer_rmse.std():,.0f}")
        k3.metric("최고 R² (fold)",    f"{outer_r2.max():.4f}")
        k4.metric("최저 R² (fold)",    f"{outer_r2.min():.4f}")

        st.markdown("---")

        # ── Fold별 R² 박스플롯 + 라인차트
        st.markdown("#### 📈 Outer Fold별 R² 분포 & Inner vs Outer 비교")

        col_a, col_b = st.columns(2)

        with col_a:
            # 박스플롯: outer R²
            fig_box, ax_box = plt.subplots(figsize=(6, 5))
            ax_box.boxplot(outer_r2, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#AEE2F7", color="#2E86AB"),
                           medianprops=dict(color="#E8505B", linewidth=2),
                           whiskerprops=dict(color="#2E86AB"),
                           capprops=dict(color="#2E86AB"),
                           flierprops=dict(marker="o", color="#E8505B", markersize=6))
            ax_box.scatter([1]*len(outer_r2), outer_r2,
                           color="#2E86AB", zorder=5, s=60, edgecolors="k", alpha=0.8)
            ax_box.set_xticks([1])
            ax_box.set_xticklabels([f"Outer R²\n(n={outer_k})"])
            ax_box.set_ylabel("R²")
            ax_box.set_title("Outer Fold R² 분포\n(박스플롯)", fontweight="bold")
            ax_box.axhline(outer_r2.mean(), color="black", linestyle="--",
                           linewidth=1, label=f"Mean={outer_r2.mean():.4f}")
            ax_box.legend(fontsize=9)
            ax_box.grid(alpha=0.3)
            st.pyplot(fig_box)
            plt.close()

        with col_b:
            # Inner vs Outer R² 라인 비교
            folds = list(range(1, outer_k + 1))
            fig_line, ax_line = plt.subplots(figsize=(6, 5))
            ax_line.plot(folds, inner_r2_all, "o-", color="#26de81",
                         linewidth=2, markersize=7, label="Inner R² (mean)")
            ax_line.plot(folds, outer_r2_list, "s--", color="#E8505B",
                         linewidth=2, markersize=7, label="Outer R²")
            ax_line.fill_between(folds, inner_r2_all, outer_r2_list,
                                 alpha=0.15, color="orange", label="Gap")
            ax_line.set_xlabel("Fold")
            ax_line.set_ylabel("R²")
            ax_line.set_title("Inner vs Outer R² 비교\n(과적합 갭 확인)", fontweight="bold")
            ax_line.legend(fontsize=9)
            ax_line.grid(alpha=0.3)
            ax_line.set_xticks(folds)
            st.pyplot(fig_line)
            plt.close()

        # ── Outer RMSE per fold 바 차트 (Plotly)
        st.markdown("---")
        st.markdown("#### 📊 Fold별 Outer RMSE")
        fig_rmse_bar = go.Figure(go.Bar(
            x=[f"Fold {i}" for i in range(1, outer_k+1)],
            y=outer_rmse_list,
            marker_color=["#E8505B" if v == max(outer_rmse_list) else "#2E86AB"
                          for v in outer_rmse_list],
            text=[f"{v:,.0f}" for v in outer_rmse_list],
            textposition="outside",
        ))
        fig_rmse_bar.add_hline(
            y=outer_rmse.mean(), line_dash="dash", line_color="black",
            annotation_text=f"Mean RMSE={outer_rmse.mean():,.0f}",
            annotation_position="top right",
        )
        fig_rmse_bar.update_layout(
            height=380, template="plotly_white",
            yaxis_title="RMSE", xaxis_title="Fold",
            title="Fold별 Outer RMSE (빨강=최대 오류 fold)",
        )
        st.plotly_chart(fig_rmse_bar, use_container_width=True)

        # ── 상세 테이블
        st.markdown("---")
        st.markdown("#### 🗂 Fold별 상세 결과 테이블")
        detail_df = pd.DataFrame(fold_details)
        gap_mean  = detail_df["Gap (Inner-Outer)"].mean()

        styled = detail_df.style.background_gradient(
            subset=["Inner R² (평균)", "Outer R²"], cmap="Blues"
        ).background_gradient(
            subset=["Gap (Inner-Outer)"], cmap="Reds"
        ).format({
            "Inner R² (평균)": "{:.4f}",
            "Outer R²":        "{:.4f}",
            "Outer RMSE":      "{:,.2f}",
            "Gap (Inner-Outer)": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True)

        if gap_mean < 0.05:
            st.success(f"✅ 평균 Inner-Outer 갭 = {gap_mean:.4f} → 과적합 위험 낮음, 일반화 성능 안정적")
        elif gap_mean < 0.15:
            st.warning(f"⚠️ 평균 Inner-Outer 갭 = {gap_mean:.4f} → 약간의 과적합 존재. 데이터 추가 권장")
        else:
            st.error(f"🚨 평균 Inner-Outer 갭 = {gap_mean:.4f} → 과적합 위험 높음. 정규화/데이터 보강 필요")

        st.markdown("""
---
**해석 가이드:**
- **Inner R² > Outer R²**: 정상 (학습 데이터에 더 잘 맞는 것은 당연)
- **갭이 작을수록** → 모델이 새 데이터에도 안정적으로 작동
- **fold 간 편차가 작을수록** → 특정 분할에 운이 작용하지 않음을 의미
""")
