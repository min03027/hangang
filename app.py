"""
한강공원 이용객 분석 대시보드
============================
EDA → t-test → VIF → 모델(pkl) → 잔차 → SHAP → Conformal → Bootstrap → Nested CV
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import shap

warnings.filterwarnings("ignore")

# ── 한글 폰트
import platform
sys_name = platform.system()
if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    try:
        fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = "NanumGothic"
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ── 페이지 설정
st.set_page_config(
    page_title="한강공원 분석 대시보드",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 900; margin: 0; letter-spacing: -1px; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
    .metric-card {
        background: white; border: 1px solid #e8ecef; border-radius: 12px;
        padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card h3    { font-size: 0.85rem; color: #666; margin: 0 0 0.3rem 0; font-weight: 400; }
    .metric-card .value{ font-size: 1.6rem; font-weight: 700; color: #1A1A2E; }
    .section-header {
        border-left: 4px solid #2E86AB; padding-left: 12px;
        margin: 2rem 0 1rem 0; font-weight: 700;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F2027 0%, #203A43 100%); }
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# pkl 모델 로드 (예측 전용 — fit 절대 호출 안 함)
# ══════════════════════════════════════════════
@st.cache_resource
def load_pkl_model():
    """model/ 폴더의 첫 번째 pkl을 로드. 이미 fitted된 모델이므로 predict만 사용."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None, None
    with open(pkl_files[0], "rb") as f:
        obj = pickle.load(f)
    return obj, os.path.basename(pkl_files[0])

pkl_model, pkl_name = load_pkl_model()


def get_pkl_feature_info(model):
    """pkl 모델이 학습 시 사용한 피처 수 / 이름 추출 시도"""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_   # int 반환
    return None


# ══════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    df = pd.read_csv(os.path.join(data_dir, "users.csv"), encoding="utf-8")
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M")

    num_cols = [
        "자전거","인라인","pm(개인형이동장치)","주요행사","마라톤","운동시설",
        "야구장","론볼링장","트랙구장","롤러장","자전거공원","외국인",
        "수상시설","수영장/물놀이장","빙상장/눈설매장","전망쉼터","캠핑장",
        "자연학습장","음악분수","키즈랜드","장미원","x게임장","자벌레",
        "달빛무지개","세빛섬","수상무대","계절,녹음수광장","천상계단",
        "피아노물길","멀티프라자","서울색공원","물빛광장","너플들판테라스",
        "골프장","여의도샛강","여의도시민 요트나루","평화공원브릿지","거울분수",
        "강변물놀이장","강변프롬나드","난지 하늘다리","갈대숲탐장로",
        "꿀벌숲","치유의숲","그라스정원","노들섬","습지생태공원",
    ]
    time_cols = ["일반이용자(아침)","일반이용자(낮)","일반이용자(저녁)"]
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
        if m in [3,4,5]: return "봄"
        elif m in [6,7,8]: return "여름"
        elif m in [9,10,11]: return "가을"
        return "겨울"
    merged["계절"] = merged["월"].apply(get_season)

    park_list = [
        "광나루한강공원","이촌한강공원","뚝섬한강공원","잠실한강공원",
        "양화한강공원","망원한강공원","반포한강공원","잠원한강공원",
        "강서한강공원","여의도한강공원","난지한강공원",
    ]
    return merged, num_cols, all_num, park_list

merged, num_cols, all_num, park_list = load_data()


# ── 유틸
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
            try:   v = variance_inflation_factor(X_c.values, i+1)
            except: v = 0
            vifs.append((col, v))
        vif_df  = pd.DataFrame(vifs, columns=["Feature","VIF"])
        max_row = vif_df.loc[vif_df["VIF"].idxmax()]
        if max_row["VIF"] <= threshold:
            break
        removed.append((max_row["Feature"], round(max_row["VIF"],1)))
        cols.remove(max_row["Feature"])
    return cols, removed

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    alpha  = (100 - ci) / 2
    return np.mean(scores), np.percentile(scores, alpha), np.percentile(scores, 100-alpha)


# ══════════════════════════════════════════════
# 지도
# ══════════════════════════════════════════════
st.markdown("## 🗺️ 한강공원 선택")
parks = {
    "강서한강공원":[37.588,126.815],"양화한강공원":[37.543,126.901],
    "난지한강공원":[37.568,126.876],"망원한강공원":[37.555,126.897],
    "여의도한강공원":[37.528,126.932],"이촌한강공원":[37.517,126.973],
    "반포한강공원":[37.510,126.995],"잠원한강공원":[37.519,127.011],
    "잠실한강공원":[37.520,127.086],"뚝섬한강공원":[37.529,127.072],
    "광나루한강공원":[37.548,127.118],
}
m_map = folium.Map(location=[37.53,126.98], zoom_start=11, tiles="CartoDB positron")
for park, coord in parks.items():
    folium.Marker(location=coord, tooltip=park, popup=park,
                  icon=folium.Icon(color="blue", icon="tree-deciduous")).add_to(m_map)
map_data = st_folium(m_map, width=1000, height=500)
selected_park = "여의도한강공원"
if map_data["last_object_clicked_popup"]:
    selected_park = map_data["last_object_clicked_popup"]
st.success(f"선택된 공원: {selected_park}")

# pkl 상태
if pkl_model is not None:
    feat_info = get_pkl_feature_info(pkl_model)
    if isinstance(feat_info, list):
        st.info(f"🤖 로드된 모델: **{pkl_name}** | 학습 피처 수: {len(feat_info)}개")
    elif isinstance(feat_info, int):
        st.info(f"🤖 로드된 모델: **{pkl_name}** | 학습 피처 수: {feat_info}개")
    else:
        st.info(f"🤖 로드된 모델: **{pkl_name}**")
else:
    st.error("⚠️ model/ 폴더에 pkl 파일이 없습니다.")

page = st.radio(
    "분석 메뉴",
    ["📊 EDA (탐색적 분석)","🔬 t-test & VIF","🤖 모델 예측 (pkl)",
     "📈 잔차 도표 & 진단","💡 SHAP 해석","🎯 Conformal Prediction",
     "📐 Bootstrap 95% CI","🔁 Nested CV"],
    horizontal=True,
)

st.markdown(f"""
<div class="main-header">
    <h1>🏞️ 한강공원 이용객 분석 대시보드</h1>
    <p>현재 선택: <strong>{selected_park}</strong> · 11개 한강공원 · ML 예측</p>
</div>
""", unsafe_allow_html=True)

leakage     = ["일반이용자(아침)","일반이용자(낮)","일반이용자(저녁)"]
feature_cols = [c for c in num_cols if c not in leakage]


# ══════════════════════════════════════════════
# 📊 EDA
# ══════════════════════════════════════════════
if page == "📊 EDA (탐색적 분석)":
    park_mean_search = merged[selected_park].mean()
    park_total       = merged["총이용객"].sum()
    park_corr        = merged[selected_park].corr(merged["총이용객"])

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [
        (c1,"총 이용객 (전체 기간)",f"{park_total:,.0f}"),
        (c2,"평균 월 이용객",f"{merged['총이용객'].mean():,.0f}"),
        (c3,f"{selected_park} 평균 검색량",f"{park_mean_search:.1f}"),
        (c4,"검색량↔이용객 상관",f"{park_corr:.3f}"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><h3>{label}</h3>'
                        f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<h3 class="section-header">시계열 추이</h3>', unsafe_allow_html=True)
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Scatter(x=merged["연월"],y=merged["총이용객"],name="총이용객",
                                line=dict(color="#2E86AB",width=2.5)), secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=merged["연월"],y=merged[selected_park],
                                name=f"{selected_park} 검색량",
                                line=dict(color="#E8505B",width=2,dash="dot")), secondary_y=True)
    fig_ts.update_layout(height=400,template="plotly_white",legend=dict(orientation="h",y=1.12))
    fig_ts.update_yaxes(title_text="총이용객", secondary_y=False)
    fig_ts.update_yaxes(title_text="검색량", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown('<h3 class="section-header">계절별 분포</h3>', unsafe_allow_html=True)
        fig_s = px.box(merged, x="계절", y="총이용객", color="계절",
                       color_discrete_sequence=["#26de81","#fd9644","#fc5c65","#4b7bec"])
        fig_s.update_layout(height=350,showlegend=False,template="plotly_white")
        st.plotly_chart(fig_s, use_container_width=True)
    with col_b:
        st.markdown('<h3 class="section-header">월별 평균</h3>', unsafe_allow_html=True)
        ma = merged.groupby("월")["총이용객"].mean().reset_index()
        fig_m = px.bar(ma, x="월", y="총이용객", color="총이용객", color_continuous_scale="Blues")
        fig_m.update_layout(height=350,template="plotly_white")
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown('<h3 class="section-header">공원별 평균 검색량 비교</h3>', unsafe_allow_html=True)
    pm = merged[park_list].mean().sort_values(ascending=True).reset_index()
    pm.columns = ["공원","평균 검색량"]
    colors = ["#E8505B" if p==selected_park else "#2E86AB" for p in pm["공원"]]
    fig_pb = go.Figure(go.Bar(x=pm["평균 검색량"],y=pm["공원"],orientation="h",marker_color=colors))
    fig_pb.update_layout(height=400,template="plotly_white")
    st.plotly_chart(fig_pb, use_container_width=True)

    st.markdown('<h3 class="section-header">공원 간 검색량 상관관계</h3>', unsafe_allow_html=True)
    corr_m = merged[park_list].corr()
    fig_h = px.imshow(corr_m, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    fig_h.update_layout(height=500,template="plotly_white")
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════
# 🔬 t-test & VIF
# ══════════════════════════════════════════════
elif page == "🔬 t-test & VIF":
    st.markdown('<h3 class="section-header">t-test 기반 피처 선택</h3>', unsafe_allow_html=True)
    st.info(f"**{selected_park}** 기준: 총이용객 중앙값으로 High/Low 그룹 분리 → 각 피처 t-test")

    features  = feature_cols + [selected_park]
    median_v  = merged["총이용객"].median()
    high      = merged[merged["총이용객"] >= median_v]
    low       = merged[merged["총이용객"] <  median_v]

    rows = []
    for col in features:
        x1 = pd.to_numeric(high[col], errors="coerce").dropna()
        x2 = pd.to_numeric(low[col],  errors="coerce").dropna()
        t, p = stats.ttest_ind(x1, x2, nan_policy="omit")
        rows.append({"피처":col,"t_stat":round(t,4),"p_value":round(p,4)})
    ttest_df = pd.DataFrame(rows).sort_values("p_value")
    ttest_df["유의"] = ttest_df["p_value"].apply(lambda p:"✅ 유의" if p<0.05 else "❌")
    sig_cols     = ttest_df[ttest_df["p_value"]<0.05]["피처"].tolist()
    no_vif_cols  = sig_cols + ([selected_park] if selected_park not in sig_cols else [])

    col1,col2 = st.columns([2,1])
    with col1:
        fig_tt = px.bar(ttest_df.head(20), x="p_value", y="피처", orientation="h",
                        color=ttest_df.head(20)["p_value"].apply(
                            lambda p:"유의 (p<0.05)" if p<0.05 else "비유의"),
                        color_discrete_map={"유의 (p<0.05)":"#E8505B","비유의":"#4b7bec"})
        fig_tt.add_vline(x=0.05,line_dash="dash",line_color="black")
        fig_tt.update_layout(height=500,template="plotly_white",showlegend=True)
        st.plotly_chart(fig_tt, use_container_width=True)
    with col2:
        st.markdown(f"**유의한 피처: {len(sig_cols)}개**")
        st.dataframe(ttest_df, height=500, use_container_width=True)

    st.markdown("---")
    st.markdown('<h3 class="section-header">Stepwise VIF 제거</h3>', unsafe_allow_html=True)
    if len(no_vif_cols) >= 2:
        X_sub = merged[no_vif_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        survived, removed = stepwise_vif(X_sub, threshold=10)
        if selected_park not in survived:
            survived.append(selected_park)
        c1,c2 = st.columns(2)
        with c1:
            st.success(f"**VIF 통과 변수: {len(survived)}개**")
            st.write(survived)
        with c2:
            if removed:
                st.warning(f"**제거된 변수: {len(removed)}개**")
                for feat,vif_val in removed:
                    st.write(f"  - {feat} (VIF={vif_val})")
            else:
                st.success("제거된 변수 없음")
    else:
        survived = no_vif_cols
        st.warning("변수 2개 미만 → VIF 적용 불가")

    st.session_state["vif_cols"]    = survived
    st.session_state["no_vif_cols"] = no_vif_cols


# ══════════════════════════════════════════════
# 🤖 모델 예측 (pkl — predict only)
# ══════════════════════════════════════════════
elif page == "🤖 모델 예측 (pkl)":
    st.markdown('<h3 class="section-header">pkl 모델 예측 (이미 학습된 모델)</h3>', unsafe_allow_html=True)

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다.")
        st.stop()

    st.info(f"📦 **{pkl_name}** — 이미 학습된 모델. predict()만 호출합니다.")

    # ── 피처 준비: pkl이 아는 피처만 사용
    X_all = preprocess(merged[feature_cols].copy())
    y     = merged["총이용객"]

    feat_info = get_pkl_feature_info(pkl_model)

    if isinstance(feat_info, list):
        # pkl이 feature_names_in_을 갖고 있으면 해당 피처만 사용
        missing = [f for f in feat_info if f not in X_all.columns]
        if missing:
            st.error(f"데이터에 없는 피처: {missing}")
            st.stop()
        X_use = X_all[feat_info]
        st.success(f"pkl 학습 피처 {len(feat_info)}개를 데이터에서 자동 매칭했습니다.")
    elif isinstance(feat_info, int):
        # 피처 수만 알 때 → 앞에서부터 해당 수만큼 사용
        if feat_info > len(X_all.columns):
            st.error(f"pkl 모델이 요구하는 피처 수({feat_info})가 데이터 피처 수({len(X_all.columns)})보다 많습니다.")
            st.stop()
        X_use = X_all.iloc[:, :feat_info]
        st.warning(f"피처 이름을 알 수 없어 앞 {feat_info}개 컬럼을 사용합니다. 순서가 학습 시와 동일해야 합니다.")
    else:
        X_use = X_all
        st.warning("피처 정보를 pkl에서 읽을 수 없어 모든 피처를 사용합니다.")

    # ── 스케일링 (별도 scaler가 없으므로 StandardScaler 적용)
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_use, y, test_size=0.2, random_state=42)
    sc_o = StandardScaler()
    Xo_tr_sc = sc_o.fit_transform(Xo_tr)
    Xo_te_sc = sc_o.transform(Xo_te)
    X_full_sc = sc_o.fit_transform(X_use)   # 전체 예측용

    # ── predict only (fit 호출 없음)
    try:
        pred_pkl    = pkl_model.predict(Xo_te_sc)
        pred_full   = pkl_model.predict(X_full_sc)
    except Exception as e:
        st.error(f"예측 오류: {e}\n\n피처 수 또는 스케일 문제일 수 있습니다. pkl 학습 당시와 동일한 전처리가 필요합니다.")
        st.stop()

    rmse = np.sqrt(mean_squared_error(yo_te, pred_pkl))
    mae  = mean_absolute_error(yo_te, pred_pkl)
    r2   = r2_score(yo_te, pred_pkl)

    k1,k2,k3 = st.columns(3)
    k1.metric("RMSE", f"{rmse:,.0f}")
    k2.metric("MAE",  f"{mae:,.0f}")
    k3.metric("R²",   f"{r2:.4f}")

    st.markdown('<h3 class="section-header">실제 vs 예측 (테스트셋)</h3>', unsafe_allow_html=True)
    idx = np.arange(len(yo_te))
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=idx, y=yo_te.values, mode="lines+markers",
                               name="실제", line=dict(color="#2E86AB",width=2)))
    fig_p.add_trace(go.Scatter(x=idx, y=pred_pkl, mode="lines+markers",
                               name=f"예측 ({pkl_name})", line=dict(color="#E8505B",width=2,dash="dot")))
    fig_p.update_layout(height=400, template="plotly_white",
                        xaxis_title="샘플 인덱스", yaxis_title="총이용객")
    st.plotly_chart(fig_p, use_container_width=True)

    # Ridge 계수 (coef_)
    try:
        coef = pkl_model.coef_
        coef_df = pd.DataFrame({"Feature":X_use.columns,"Coefficient":coef})
        coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("AbsCoef", ascending=False).head(15)
        st.markdown('<h3 class="section-header">Ridge 계수 (|coef| 기준 Top 15)</h3>', unsafe_allow_html=True)
        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                          color="Coefficient", color_continuous_scale="RdBu",
                          color_continuous_midpoint=0)
        fig_coef.update_layout(height=450, template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_coef, use_container_width=True)
    except Exception:
        pass

    # 세션 저장
    st.session_state.update({
        "pkl_model":  pkl_model,
        "Xo_tr_sc":   Xo_tr_sc,
        "Xo_te_sc":   Xo_te_sc,
        "yo_te":      yo_te,
        "yo_tr":      yo_tr,
        "X_use":      X_use,
        "Xo_tr":      Xo_tr,
        "Xo_te":      Xo_te,
        "sc_o":       sc_o,
        "pred_pkl":   pred_pkl,
        "X_full_sc":  X_full_sc,
        "y":          y,
    })
    st.success("✅ 세션 저장 완료 — 잔차/SHAP/Bootstrap/NestedCV 탭에서 이 결과를 사용합니다.")


# ══════════════════════════════════════════════
# 📈 잔차 도표
# ══════════════════════════════════════════════
elif page == "📈 잔차 도표 & 진단":
    if "pred_pkl" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 예측 (pkl)** 탭을 실행해주세요.")
        st.stop()

    yo_te    = st.session_state["yo_te"]
    pred_pkl = st.session_state["pred_pkl"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    yo_tr    = st.session_state["yo_tr"]
    residuals = yo_te.values - pred_pkl

    st.markdown(f'<h3 class="section-header">잔차 도표 4종 · {pkl_name}</h3>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2,2, figsize=(12,9))

    axes[0,0].scatter(pred_pkl, residuals, alpha=0.6, edgecolors="k", s=40, color="#2E86AB")
    axes[0,0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0,0].set_xlabel("Fitted Values"); axes[0,0].set_ylabel("Residuals")
    axes[0,0].set_title("① Residuals vs Fitted", fontweight="bold")

    probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("② Normal Q-Q Plot", fontweight="bold")
    axes[0,1].get_lines()[1].set_color("red")

    axes[1,0].hist(residuals, bins=12, edgecolor="black", alpha=0.7, color="#2E86AB")
    axes[1,0].set_xlabel("Residuals"); axes[1,0].set_ylabel("Frequency")
    axes[1,0].set_title("③ Histogram of Residuals", fontweight="bold")

    axes[1,1].plot(range(len(residuals)), residuals, "o-", markersize=4, alpha=0.7, color="#2E86AB")
    axes[1,1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1,1].set_xlabel("Observation Order"); axes[1,1].set_ylabel("Residuals")
    axes[1,1].set_title("④ Residuals vs Order", fontweight="bold")

    plt.suptitle(f"Residual Plots — {pkl_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    stat_sw, p_sw = stats.shapiro(residuals)
    if p_sw > 0.05:
        st.success(f"Shapiro-Wilk: p={p_sw:.4f} → 잔차가 정규분포를 따름")
    else:
        st.warning(f"Shapiro-Wilk: p={p_sw:.4f} → 정규분포에서 벗어남 (소표본에서 흔함)")


# ══════════════════════════════════════════════
# 💡 SHAP
# ══════════════════════════════════════════════
elif page == "💡 SHAP 해석":
    if "pred_pkl" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 예측 (pkl)** 탭을 실행해주세요.")
        st.stop()

    st.markdown('<h3 class="section-header">SHAP Summary Plot</h3>', unsafe_allow_html=True)
    st.info("각 피처가 개별 예측에 얼마나 기여했는지 시각화합니다.")

    model_s  = st.session_state["pkl_model"]
    Xo_tr_sc = st.session_state["Xo_tr_sc"]
    Xo_te_sc = st.session_state["Xo_te_sc"]
    Xo_te    = st.session_state["Xo_te"]
    X_use    = st.session_state["X_use"]

    est_shap = model_s
    if hasattr(model_s, "named_steps"):
        est_shap = list(model_s.named_steps.values())[-1]

    with st.spinner("SHAP 값 계산 중..."):
        try:
            explainer   = shap.Explainer(est_shap, Xo_tr_sc)
            shap_values = explainer(Xo_te_sc)
        except Exception:
            explainer   = shap.KernelExplainer(est_shap.predict, Xo_tr_sc[:50])
            shap_values = explainer.shap_values(Xo_te_sc)

    fig_shap, _ = plt.subplots(figsize=(10,8))
    shap.summary_plot(shap_values, Xo_te,
                      feature_names=X_use.columns.tolist(), show=False)
    st.pyplot(fig_shap); plt.close()

    st.markdown("---")
    st.markdown('<h3 class="section-header">SHAP 평균 기여도 (Bar)</h3>', unsafe_allow_html=True)
    fig_b2, _ = plt.subplots(figsize=(10,6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    st.pyplot(fig_b2); plt.close()


# ══════════════════════════════════════════════
# 🎯 Conformal Prediction
# ══════════════════════════════════════════════
elif page == "🎯 Conformal Prediction":
    if "pred_pkl" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 예측 (pkl)** 탭을 실행해주세요.")
        st.stop()

    st.markdown('<h3 class="section-header">Conformal Prediction (예측 구간)</h3>', unsafe_allow_html=True)
    st.markdown("""
**원리:** Calibration set 잔차 분포 → Test 예측 구간 부여
- 예측값 ± q_hat (잔차 (1-α) 백분위수)
- α = 0.1 → 목표 커버리지 90%
""")

    alpha_cp = st.slider("α (1 - 커버리지)", 0.05, 0.30, 0.10, 0.05)

    # pkl predict only: calibration/test split 후 잔차로 q_hat 계산
    X_use  = st.session_state["X_use"]
    y      = st.session_state["y"]
    sc_o   = st.session_state["sc_o"]
    model_cp = st.session_state["pkl_model"]

    X_tr, X_temp, y_tr, y_temp = train_test_split(X_use, y, test_size=0.4, random_state=42)
    X_cal, X_te, y_cal, y_te   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # scaler는 전체 훈련셋 기준으로 fit
    sc_cp = StandardScaler()
    sc_cp.fit(X_tr)
    X_cal_sc = sc_cp.transform(X_cal)
    X_te_sc  = sc_cp.transform(X_te)

    try:
        cal_pred = model_cp.predict(X_cal_sc)
        te_pred  = model_cp.predict(X_te_sc)
    except Exception as e:
        st.error(f"예측 오류: {e}")
        st.stop()

    cal_resid = np.abs(y_cal.values - cal_pred)
    q_hat     = np.quantile(cal_resid, 1 - alpha_cp)
    lower     = te_pred - q_hat
    upper     = te_pred + q_hat
    coverage  = np.mean((y_te.values >= lower) & (y_te.values <= upper))

    c1,c2,c3 = st.columns(3)
    c1.metric("q_hat (구간 반폭)",  f"{q_hat:,.0f}")
    c2.metric("목표 커버리지",       f"{(1-alpha_cp)*100:.0f}%")
    c3.metric("실제 커버리지",       f"{coverage*100:.1f}%")

    idx = np.arange(len(y_te))
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(x=idx,y=y_te.values,mode="lines+markers",
                                name="Actual",line=dict(color="#2E86AB")))
    fig_cp.add_trace(go.Scatter(x=idx,y=te_pred,mode="lines+markers",
                                name="Predicted",line=dict(color="#E8505B")))
    fig_cp.add_trace(go.Scatter(
        x=np.concatenate([idx,idx[::-1]]),
        y=np.concatenate([upper,lower[::-1]]),
        fill="toself", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int((1-alpha_cp)*100)}% Interval",
    ))
    fig_cp.update_layout(height=450,template="plotly_white")
    st.plotly_chart(fig_cp, use_container_width=True)


# ══════════════════════════════════════════════
# 📐 Bootstrap 95% CI
# ══════════════════════════════════════════════
elif page == "📐 Bootstrap 95% CI":
    if "pred_pkl" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 예측 (pkl)** 탭을 실행해주세요.")
        st.stop()

    st.markdown('<h3 class="section-header">Bootstrap 95% 신뢰구간 (성능 지표)</h3>', unsafe_allow_html=True)
    st.markdown("""
**원리:** 테스트셋에서 복원 추출(1000회) → 매 회 RMSE/MAE/R² 계산 → 2.5th~97.5th 백분위수를 CI로 사용

- 단일 분할의 운에 의존하지 않고 **지표의 안정성(변동폭)** 을 정량화
- CI 폭이 좁을수록 모델 성능이 일관적
""")

    n_boot   = st.slider("Bootstrap 반복 횟수", 500, 3000, 1000, 500)
    yo_te    = st.session_state["yo_te"]
    pred_pkl = st.session_state["pred_pkl"]
    y_true_arr = yo_te.values

    rmse_fn = lambda yt,yp: np.sqrt(mean_squared_error(yt,yp))
    mae_fn  = lambda yt,yp: mean_absolute_error(yt,yp)
    r2_fn   = lambda yt,yp: r2_score(yt,yp)

    with st.spinner(f"Bootstrap {n_boot}회 반복 중..."):
        rmse_mean,rmse_lo,rmse_hi = bootstrap_ci(y_true_arr, pred_pkl, rmse_fn, n_boot)
        mae_mean, mae_lo, mae_hi  = bootstrap_ci(y_true_arr, pred_pkl, mae_fn,  n_boot)
        r2_mean,  r2_lo,  r2_hi   = bootstrap_ci(y_true_arr, pred_pkl, r2_fn,   n_boot)

        rng = np.random.RandomState(42)
        n   = len(y_true_arr)
        rmse_dist,mae_dist,r2_dist = [],[],[]
        for _ in range(n_boot):
            idx_b = rng.randint(0,n,n)
            try:
                rmse_dist.append(rmse_fn(y_true_arr[idx_b], pred_pkl[idx_b]))
                mae_dist.append(mae_fn(y_true_arr[idx_b],   pred_pkl[idx_b]))
                r2_dist.append(r2_fn(y_true_arr[idx_b],     pred_pkl[idx_b]))
            except Exception:
                pass
        rmse_dist = np.array(rmse_dist)
        mae_dist  = np.array(mae_dist)
        r2_dist   = np.array(r2_dist)

    k1,k2,k3 = st.columns(3)
    k1.metric("RMSE (mean)", f"{rmse_mean:,.0f}", f"95% CI: [{rmse_lo:,.0f} ~ {rmse_hi:,.0f}]")
    k2.metric("MAE (mean)",  f"{mae_mean:,.0f}",  f"95% CI: [{mae_lo:,.0f} ~ {mae_hi:,.0f}]")
    k3.metric("R² (mean)",   f"{r2_mean:.4f}",    f"95% CI: [{r2_lo:.4f} ~ {r2_hi:.4f}]")

    st.markdown("---")
    st.markdown("#### 📈 Bootstrap 분포 (히스토그램 + 95% CI 음영)")

    fig_boot, axes_b = plt.subplots(1,3, figsize=(15,5))

    def plot_boot(ax, dist, lo, hi, mean_v, label, color):
        ax.hist(dist, bins=40, color=color, alpha=0.65, edgecolor="none")
        ax.axvspan(lo, hi, alpha=0.22, color=color, label="95% CI")
        ax.axvline(mean_v, color="black",   linestyle="--", linewidth=1.5, label=f"Mean={mean_v:.3g}")
        ax.axvline(lo,     color="crimson", linestyle=":",  linewidth=1.2, label=f"Lo={lo:.3g}")
        ax.axvline(hi,     color="crimson", linestyle=":",  linewidth=1.2, label=f"Hi={hi:.3g}")
        ax.set_xlabel(label,fontsize=11); ax.set_ylabel("Frequency",fontsize=10)
        ax.set_title(f"Bootstrap — {label}", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plot_boot(axes_b[0], rmse_dist, rmse_lo, rmse_hi, rmse_mean, "RMSE", "#2E86AB")
    plot_boot(axes_b[1], mae_dist,  mae_lo,  mae_hi,  mae_mean,  "MAE",  "#26de81")
    plot_boot(axes_b[2], r2_dist,   r2_lo,   r2_hi,   r2_mean,   "R²",   "#E8505B")

    plt.suptitle(f"Bootstrap {n_boot}회 · {pkl_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_boot); plt.close()

    st.markdown("---")
    st.markdown("#### 📊 신뢰구간 폭(Width) 요약")
    ci_df = pd.DataFrame({
        "지표":  ["RMSE","MAE","R²"],
        "Mean":  [rmse_mean, mae_mean, r2_mean],
        "CI_lo": [rmse_lo, mae_lo, r2_lo],
        "CI_hi": [rmse_hi, mae_hi, r2_hi],
        "Width": [rmse_hi-rmse_lo, mae_hi-mae_lo, r2_hi-r2_lo],
    })
    fig_ci = go.Figure()
    colors_ci = ["#2E86AB","#26de81","#E8505B"]
    for i, row in ci_df.iterrows():
        fig_ci.add_trace(go.Bar(
            name=row["지표"], x=[row["지표"]], y=[row["Width"]],
            marker_color=colors_ci[i],
            text=f"±{row['Width']:.3g}", textposition="outside",
            error_y=dict(type="data",symmetric=False,
                         array=[row["CI_hi"]-row["Mean"]],
                         arrayminus=[row["Mean"]-row["CI_lo"]]),
        ))
    fig_ci.update_layout(height=380, template="plotly_white", showlegend=False,
                         title="95% CI 폭 (좁을수록 안정적)", yaxis_title="CI Width")
    st.plotly_chart(fig_ci, use_container_width=True)
    st.dataframe(ci_df.style.format({"Mean":"{:.4g}","CI_lo":"{:.4g}",
                                      "CI_hi":"{:.4g}","Width":"{:.4g}"}),
                 use_container_width=True)


# ══════════════════════════════════════════════
# 🔁 Nested CV
# ══════════════════════════════════════════════
elif page == "🔁 Nested CV":
    st.markdown('<h3 class="section-header">Nested Cross-Validation (신뢰도 향상)</h3>', unsafe_allow_html=True)
    st.markdown("""
**구조:**
- **Outer loop (K=5):** 모델 성능의 불편 추정 → fold별 R² / RMSE
- **Inner loop (K=3):** pkl 고정 파라미터로 검증 성능 추정
- Inner-Outer 갭이 작을수록 → **과적합 없이 일반화**

> ⚠️ pkl 모델은 이미 학습된 상태이므로, **각 fold에서 새 Ridge (동일 파라미터) 를 재학습**하여 Nested CV를 수행합니다.
""")

    if pkl_model is None:
        st.error("model/ 폴더에 pkl 파일이 없습니다.")
        st.stop()

    # pkl의 alpha 추출 (Ridge인 경우)
    pkl_alpha = getattr(pkl_model, "alpha", 1.0)
    st.info(f"pkl에서 읽은 Ridge alpha: **{pkl_alpha}** → 동일 파라미터로 Nested CV 수행")

    outer_k = st.slider("Outer Fold 수 (K)", 3, 10, 5)
    inner_k = st.slider("Inner Fold 수 (K)", 2,  5, 3)

    if "X_use" not in st.session_state:
        st.warning("⬅️ 먼저 **모델 예측 (pkl)** 탭을 실행해주세요.")
        st.stop()

    if st.button("🚀 Nested CV 실행", type="primary"):
        X_ncv = preprocess(st.session_state["X_use"].copy())
        y_ncv = st.session_state["y"]

        sc_ncv = StandardScaler()
        X_sc   = sc_ncv.fit_transform(X_ncv)
        y_arr  = y_ncv.values

        outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=42)

        outer_r2_list, outer_rmse_list = [], []
        inner_r2_all   = []
        fold_details   = []

        with st.spinner(f"Nested CV 실행 중... (outer {outer_k} × inner {inner_k})"):
            for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X_sc), 1):
                X_tr_o, X_te_o = X_sc[tr_idx], X_sc[te_idx]
                y_tr_o, y_te_o = y_arr[tr_idx], y_arr[te_idx]

                # Inner CV
                inner_scores = []
                for tr_i, val_i in inner_cv.split(X_tr_o):
                    m_in = Ridge(alpha=pkl_alpha)
                    m_in.fit(X_tr_o[tr_i], y_tr_o[tr_i])
                    p_in = m_in.predict(X_tr_o[val_i])
                    inner_scores.append(r2_score(y_tr_o[val_i], p_in))
                inner_mean_r2 = np.mean(inner_scores)
                inner_r2_all.append(inner_mean_r2)

                # Outer
                m_out = Ridge(alpha=pkl_alpha)
                m_out.fit(X_tr_o, y_tr_o)
                p_out    = m_out.predict(X_te_o)
                r2_out   = r2_score(y_te_o, p_out)
                rmse_out = np.sqrt(mean_squared_error(y_te_o, p_out))
                outer_r2_list.append(r2_out)
                outer_rmse_list.append(rmse_out)

                fold_details.append({
                    "Fold": fold_i,
                    "Inner R² (평균)":    round(inner_mean_r2, 4),
                    "Outer R²":           round(r2_out,        4),
                    "Outer RMSE":         round(rmse_out,      2),
                    "Gap (Inner-Outer)":  round(inner_mean_r2 - r2_out, 4),
                })

        outer_r2   = np.array(outer_r2_list)
        outer_rmse = np.array(outer_rmse_list)

        # KPI
        st.markdown("#### 📊 Nested CV 결과 요약")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Outer R² (mean)", f"{outer_r2.mean():.4f}", f"± {outer_r2.std():.4f}")
        k2.metric("Outer RMSE (mean)", f"{outer_rmse.mean():,.0f}", f"± {outer_rmse.std():,.0f}")
        k3.metric("최고 R² (fold)", f"{outer_r2.max():.4f}")
        k4.metric("최저 R² (fold)", f"{outer_r2.min():.4f}")

        st.markdown("---")
        st.markdown("#### 📈 Outer Fold R² 분포 & Inner vs Outer 비교")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_box, ax_box = plt.subplots(figsize=(6,5))
            ax_box.boxplot(outer_r2, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#AEE2F7",color="#2E86AB"),
                           medianprops=dict(color="#E8505B",linewidth=2),
                           whiskerprops=dict(color="#2E86AB"),
                           capprops=dict(color="#2E86AB"),
                           flierprops=dict(marker="o",color="#E8505B",markersize=6))
            ax_box.scatter([1]*len(outer_r2), outer_r2,
                           color="#2E86AB",zorder=5,s=60,edgecolors="k",alpha=0.8)
            ax_box.set_xticks([1]); ax_box.set_xticklabels([f"Outer R² (n={outer_k})"])
            ax_box.set_ylabel("R²")
            ax_box.set_title("Outer Fold R² 분포", fontweight="bold")
            ax_box.axhline(outer_r2.mean(), color="black", linestyle="--", linewidth=1,
                           label=f"Mean={outer_r2.mean():.4f}")
            ax_box.legend(fontsize=9); ax_box.grid(alpha=0.3)
            st.pyplot(fig_box); plt.close()

        with col_b:
            folds = list(range(1, outer_k+1))
            fig_line, ax_line = plt.subplots(figsize=(6,5))
            ax_line.plot(folds, inner_r2_all, "o-", color="#26de81",
                         linewidth=2, markersize=7, label="Inner R² (mean)")
            ax_line.plot(folds, outer_r2_list, "s--", color="#E8505B",
                         linewidth=2, markersize=7, label="Outer R²")
            ax_line.fill_between(folds, inner_r2_all, outer_r2_list,
                                 alpha=0.18, color="orange", label="Gap")
            ax_line.set_xlabel("Fold"); ax_line.set_ylabel("R²")
            ax_line.set_title("Inner vs Outer R² 비교\n(과적합 갭 확인)", fontweight="bold")
            ax_line.legend(fontsize=9); ax_line.grid(alpha=0.3); ax_line.set_xticks(folds)
            st.pyplot(fig_line); plt.close()

        st.markdown("---")
        st.markdown("#### 📊 Fold별 Outer RMSE")
        fig_rmse = go.Figure(go.Bar(
            x=[f"Fold {i}" for i in range(1, outer_k+1)],
            y=outer_rmse_list,
            marker_color=["#E8505B" if v==max(outer_rmse_list) else "#2E86AB" for v in outer_rmse_list],
            text=[f"{v:,.0f}" for v in outer_rmse_list],
            textposition="outside",
        ))
        fig_rmse.add_hline(y=outer_rmse.mean(), line_dash="dash", line_color="black",
                           annotation_text=f"Mean={outer_rmse.mean():,.0f}",
                           annotation_position="top right")
        fig_rmse.update_layout(height=380, template="plotly_white",
                               yaxis_title="RMSE", xaxis_title="Fold",
                               title="Fold별 Outer RMSE (빨강=최대 오류 fold)")
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🗂 Fold별 상세 결과")
        det_df   = pd.DataFrame(fold_details)
        gap_mean = det_df["Gap (Inner-Outer)"].mean()
        styled = det_df.style\
            .background_gradient(subset=["Inner R² (평균)","Outer R²"], cmap="Blues")\
            .background_gradient(subset=["Gap (Inner-Outer)"], cmap="Reds")\
            .format({"Inner R² (평균)":"{:.4f}","Outer R²":"{:.4f}",
                     "Outer RMSE":"{:,.2f}","Gap (Inner-Outer)":"{:.4f}"})
        st.dataframe(styled, use_container_width=True)

        if gap_mean < 0.05:
            st.success(f"✅ 평균 Inner-Outer 갭 = {gap_mean:.4f} → 과적합 위험 낮음, 일반화 성능 안정적")
        elif gap_mean < 0.15:
            st.warning(f"⚠️ 평균 Inner-Outer 갭 = {gap_mean:.4f} → 약간의 과적합. 데이터 추가 권장")
        else:
            st.error(f"🚨 평균 Inner-Outer 갭 = {gap_mean:.4f} → 과적합 위험 높음. 정규화/데이터 보강 필요")

        st.markdown("""
---
**해석 가이드:**
- **Inner R² > Outer R²** → 정상 (학습 데이터에 더 잘 맞는 것은 당연)
- **갭이 작을수록** → 새 데이터에도 안정적으로 작동
- **fold 간 편차가 작을수록** → 특정 분할에 운이 작용하지 않음
""")
