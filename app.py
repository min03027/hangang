"""
한강공원 이용객 분석 대시보드
- 단일 PKL 모델 기반
- Nested Cross Validation
- Bootstrap 95% Confidence Interval
- SHAP 해석
- Residual Diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import platform

from scipy import stats
from scipy.stats import probplot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.utils import resample

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================================================
# 한글 폰트
# =========================================================

sys_name = platform.system()

if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="한강공원 분석 대시보드",
    page_icon="🏞️",
    layout="wide",
)

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.main-header{
    background: linear-gradient(135deg,#0F2027,#203A43,#2C5364);
    padding:2rem;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:2rem;
}

.metric-card{
    background:white;
    border-radius:12px;
    padding:1rem;
    text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

.section-header{
    border-left:5px solid #2E86AB;
    padding-left:10px;
    margin-top:2rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 데이터 로드
# =========================================================

@st.cache_data
def load_data():

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    df = pd.read_csv(
        os.path.join(data_dir, "users.csv"),
        encoding="utf-8"
    )

    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M")

    num_cols = [
        "자전거", "인라인", "pm(개인형이동장치)",
        "주요행사", "마라톤", "운동시설",
        "야구장", "론볼링장", "트랙구장",
        "롤러장", "자전거공원", "외국인",
        "수상시설", "수영장/물놀이장",
        "빙상장/눈설매장", "전망쉼터",
        "캠핑장", "자연학습장", "음악분수",
        "키즈랜드", "장미원", "x게임장",
        "자벌레", "달빛무지개", "세빛섬",
        "수상무대", "계절,녹음수광장",
        "천상계단", "피아노물길",
        "멀티프라자", "서울색공원",
        "물빛광장", "너플들판테라스",
        "골프장", "여의도샛강",
        "여의도시민 요트나루",
        "평화공원브릿지", "거울분수",
        "강변물놀이장", "강변프롬나드",
        "난지 하늘다리", "갈대숲탐장로",
        "꿀벌숲", "치유의숲",
        "그라스정원", "노들섬",
        "습지생태공원",
    ]

    time_cols = [
        "일반이용자(아침)",
        "일반이용자(낮)",
        "일반이용자(저녁)"
    ]

    all_num = time_cols + num_cols

    for c in all_num:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            ).fillna(0)

    df["총이용객"] = (
        df["일반이용자(아침)"]
        + df["일반이용자(낮)"]
        + df["일반이용자(저녁)"]
    )

    monthly = (
        df.groupby("연월")[all_num + ["총이용객"]]
        .sum()
        .reset_index()
    )

    monthly["연월"] = monthly["연월"].dt.to_timestamp()

    trend = pd.read_excel(
        os.path.join(data_dir, "trend.xlsx")
    )

    trend.rename(columns={"날짜": "연월"}, inplace=True)
    trend["연월"] = pd.to_datetime(trend["연월"])

    merged = pd.merge(
        monthly,
        trend,
        on="연월",
        how="inner"
    )

    return merged, num_cols


merged, num_cols = load_data()

# =========================================================
# 전처리
# =========================================================

def preprocess(X):

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X

# =========================================================
# 모델 로드
# =========================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "model.pkl"
    )

    model = joblib.load(model_path)

    return model

# =========================================================
# 헤더
# =========================================================

st.markdown(
    """
<div class="main-header">
<h1>🏞️ 한강공원 이용객 분석 대시보드</h1>
<p>PKL 단일 모델 · Nested CV · Bootstrap CI</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 메뉴
# =========================================================

page = st.radio(
    "메뉴 선택",
    [
        "📊 EDA",
        "🤖 모델 평가",
        "📈 잔차 분석",
        "💡 SHAP 해석",
    ],
    horizontal=True
)

# =========================================================
# 공통 변수
# =========================================================

leakage = [
    "일반이용자(아침)",
    "일반이용자(낮)",
    "일반이용자(저녁)"
]

feature_cols = [
    c for c in num_cols
    if c not in leakage
]

X = preprocess(
    merged[feature_cols].copy()
)

y = merged["총이용객"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# 📊 EDA
# =========================================================

if page == "📊 EDA":

    st.markdown(
        '<h3 class="section-header">기초 통계</h3>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "총 이용객",
        f"{merged['총이용객'].sum():,.0f}"
    )

    c2.metric(
        "평균 이용객",
        f"{merged['총이용객'].mean():,.0f}"
    )

    c3.metric(
        "데이터 개수",
        len(merged)
    )

    # 시계열
    fig_ts = px.line(
        merged,
        x="연월",
        y="총이용객",
        title="월별 총 이용객 추이"
    )

    st.plotly_chart(
        fig_ts,
        use_container_width=True
    )

    # 상관관계
    corr = merged[feature_cols + ["총이용객"]].corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

# =========================================================
# 🤖 모델 평가
# =========================================================

elif page == "🤖 모델 평가":

    st.markdown(
        '<h3 class="section-header">Nested Cross Validation</h3>',
        unsafe_allow_html=True
    )

    model = load_model()

    outer_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    inner_cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="r2",
        n_jobs=-1,
    )

    nested_scores = cross_val_score(
        search,
        X_scaled,
        y,
        cv=outer_cv,
        scoring="r2",
        n_jobs=-1,
    )

    mean_score = nested_scores.mean()
    std_score = nested_scores.std()

    c1, c2 = st.columns(2)

    c1.metric(
        "평균 R²",
        f"{mean_score:.4f}"
    )

    c2.metric(
        "표준편차",
        f"{std_score:.4f}"
    )

    # Fold 그래프
    fold_df = pd.DataFrame({
        "Fold": np.arange(
            1,
            len(nested_scores) + 1
        ),
        "R2": nested_scores
    })

    fig_fold = px.line(
        fold_df,
        x="Fold",
        y="R2",
        markers=True,
        title="Nested CV Fold Score"
    )

    st.plotly_chart(
        fig_fold,
        use_container_width=True
    )

    # =====================================================
    # Bootstrap
    # =====================================================

    st.markdown(
        '<h3 class="section-header">Bootstrap 95% CI</h3>',
        unsafe_allow_html=True
    )

    n_bootstrap = 1000

    bootstrap_scores = []

    progress = st.progress(0)

    for i in range(n_bootstrap):

        X_resampled, y_resampled = resample(
            X_scaled,
            y,
            replace=True,
        )

        model.fit(
            X_resampled,
            y_resampled
        )

        pred = model.predict(X_scaled)

        score = r2_score(y, pred)

        bootstrap_scores.append(score)

        progress.progress((i + 1) / n_bootstrap)

    lower = np.percentile(
        bootstrap_scores,
        2.5
    )

    upper = np.percentile(
        bootstrap_scores,
        97.5
    )

    st.success(
        f"""
        95% Confidence Interval:
        [{lower:.4f}, {upper:.4f}]
        """
    )

    # 분포 그래프
    fig_boot = px.histogram(
        x=bootstrap_scores,
        nbins=40,
        title="Bootstrap R² Distribution"
    )

    fig_boot.add_vline(
        x=lower,
        line_dash="dash",
        line_color="red"
    )

    fig_boot.add_vline(
        x=upper,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(
        fig_boot,
        use_container_width=True
    )

# =========================================================
# 📈 잔차 분석
# =========================================================

elif page == "📈 잔차 분석":

    st.markdown(
        '<h3 class="section-header">Residual Diagnostics</h3>',
        unsafe_allow_html=True
    )

    model = load_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    residuals = y_test - pred

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    # Residual vs fitted
    axes[0, 0].scatter(
        pred,
        residuals
    )

    axes[0, 0].axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    axes[0, 0].set_title(
        "Residuals vs Fitted"
    )

    # QQ plot
    probplot(
        residuals,
        dist="norm",
        plot=axes[0, 1]
    )

    axes[0, 1].set_title(
        "QQ Plot"
    )

    # Histogram
    axes[1, 0].hist(
        residuals,
        bins=10
    )

    axes[1, 0].set_title(
        "Residual Histogram"
    )

    # Residual order
    axes[1, 1].plot(
        residuals.values
    )

    axes[1, 1].axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    axes[1, 1].set_title(
        "Residual Order"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # 성능
    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    r2 = r2_score(
        y_test,
        pred
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", f"{rmse:,.0f}")
    c2.metric("MAE", f"{mae:,.0f}")
    c3.metric("R²", f"{r2:.4f}")

    # 정규성
    sw_stat, sw_p = stats.shapiro(residuals)

    if sw_p > 0.05:
        st.success(
            f"Shapiro-Wilk p={sw_p:.4f} → 정규성 만족"
        )
    else:
        st.warning(
            f"Shapiro-Wilk p={sw_p:.4f} → 정규성 미흡"
        )

# =========================================================
# 💡 SHAP
# =========================================================

elif page == "💡 SHAP 해석":

    st.markdown(
        '<h3 class="section-header">SHAP Feature Importance</h3>',
        unsafe_allow_html=True
    )

    model = load_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    with st.spinner("SHAP 계산 중..."):

        explainer = shap.Explainer(
            model,
            X_train
        )

        shap_values = explainer(X_test)

    # Summary Plot
    fig_shap = plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        show=False
    )

    st.pyplot(fig_shap)

    plt.close()

    # Bar Plot
    fig_bar = plt.figure(
        figsize=(10, 6)
    )

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False
    )

    st.pyplot(fig_bar)

    plt.close()
