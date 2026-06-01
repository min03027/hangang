"""
한강공원 이용객 분석 대시보드 — Apple-inspired retheme
- design.md 토큰 충실 적용 (SF Pro / Action Blue / 전면-블리드 타일 / 알터네이팅 light↔dark)
- 이모지 미사용, 인라인 SVG 아이콘 사용
- 단일 액션 블루, 단일 product-shadow, 가벼운 hairline, pill CTA
"""

from __future__ import annotations

import glob
import os
import pickle
import platform
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ── 한글 폰트 (fallback)
sys_name = platform.system()
if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="한강공원 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Design tokens — design.md
# ─────────────────────────────────────────────────────────────
TOK = {
    # Brand & accent
    "primary":          "#0066cc",
    "primary_focus":    "#0071e3",
    "primary_on_dark":  "#2997ff",
    # Surface
    "canvas":           "#ffffff",
    "parchment":        "#f5f5f7",
    "pearl":            "#fafafc",
    "tile1":            "#272729",
    "tile2":            "#2a2a2c",
    "tile3":            "#252527",
    "black":            "#000000",
    # Text
    "ink":              "#1d1d1f",
    "on_dark":          "#ffffff",
    "muted_dark":       "#cccccc",
    "ink_80":           "#333333",
    "ink_48":           "#7a7a7a",
    # Hairlines
    "divider_soft":     "rgba(0,0,0,0.04)",
    "hairline":         "#e0e0e0",
    # Product shadow (the ONE shadow)
    "product_shadow":   "rgba(0, 0, 0, 0.22) 3px 5px 30px 0",
}

PLOT_PALETTE = [TOK["primary"], TOK["ink"], TOK["primary_on_dark"], TOK["ink_48"], "#86868b"]

# ─────────────────────────────────────────────────────────────
# Global CSS — Apple 시스템
# ─────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
:root {{
  --primary: {TOK['primary']};
  --primary-focus: {TOK['primary_focus']};
  --primary-on-dark: {TOK['primary_on_dark']};
  --canvas: {TOK['canvas']};
  --parchment: {TOK['parchment']};
  --pearl: {TOK['pearl']};
  --tile1: {TOK['tile1']};
  --tile2: {TOK['tile2']};
  --tile3: {TOK['tile3']};
  --ink: {TOK['ink']};
  --on-dark: {TOK['on_dark']};
  --muted-dark: {TOK['muted_dark']};
  --ink-80: {TOK['ink_80']};
  --ink-48: {TOK['ink_48']};
  --hairline: {TOK['hairline']};
  --divider-soft: {TOK['divider_soft']};
  --product-shadow: {TOK['product_shadow']};
}}

/* Reset Streamlit chrome */
html, body, [class*="css"]  {{
  font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Inter", "Noto Sans KR", system-ui, sans-serif;
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.01em;
}}

.stApp {{
  background: var(--parchment);
}}

#MainMenu, footer, header[data-testid="stHeader"] {{
  visibility: hidden;
  height: 0;
}}

.block-container {{
  padding: 0 !important;
  max-width: 100% !important;
}}

/* ── Display typography ───────────────────────────────────── */
.h-hero {{
  font-family: "SF Pro Display", -apple-system, "Inter", system-ui, sans-serif;
  font-size: 56px;
  font-weight: 600;
  line-height: 1.07;
  letter-spacing: -0.028em;
  margin: 0;
}}
.h-display {{
  font-family: "SF Pro Display", -apple-system, "Inter", system-ui, sans-serif;
  font-size: 40px;
  font-weight: 600;
  line-height: 1.10;
  letter-spacing: -0.02em;
  margin: 0;
}}
.h-section {{
  font-family: "SF Pro Display", -apple-system, "Inter", system-ui, sans-serif;
  font-size: 34px;
  font-weight: 600;
  line-height: 1.18;
  letter-spacing: -0.022em;
  margin: 0 0 24px 0;
}}
.lead {{
  font-size: 24px;
  font-weight: 400;
  line-height: 1.33;
  letter-spacing: -0.005em;
  color: var(--ink-80);
  margin: 12px 0 0 0;
}}
.lead-on-dark {{ color: var(--muted-dark); }}
.tagline {{
  font-size: 21px;
  font-weight: 600;
  line-height: 1.19;
  letter-spacing: 0.011em;
  margin: 0;
}}
.body-strong {{ font-size: 17px; font-weight: 600; letter-spacing: -0.022em; }}
.body-text   {{ font-size: 17px; font-weight: 400; letter-spacing: -0.022em; line-height: 1.47; }}
.caption     {{ font-size: 14px; font-weight: 400; letter-spacing: -0.016em; color: var(--ink-48); }}
.fineprint   {{ font-size: 12px; font-weight: 400; letter-spacing: -0.01em;  color: var(--ink-48); }}

/* ── Global nav ───────────────────────────────────────────── */
.global-nav {{
  background: {TOK['black']};
  color: var(--on-dark);
  padding: 12px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  letter-spacing: -0.01em;
}}
.global-nav .nav-links {{ display:flex; gap: 22px; align-items:center; }}
.global-nav a {{ color: var(--on-dark); text-decoration: none; opacity: 0.88; }}
.global-nav a:hover {{ opacity: 1; }}

/* ── Sub-nav frosted ──────────────────────────────────────── */
.sub-nav {{
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(245,245,247,0.80);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--divider-soft);
  padding: 14px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.sub-nav .label {{ font-size: 21px; font-weight: 600; letter-spacing: 0.011em; }}
.sub-nav .right {{ display:flex; gap: 18px; align-items:center; }}
.sub-nav .right a {{ font-size: 14px; color: var(--ink); text-decoration:none; letter-spacing:-0.016em; }}
.sub-nav .right a:hover {{ color: var(--primary); }}

/* ── Tiles ────────────────────────────────────────────────── */
.tile {{
  padding: 80px 32px;
  text-align: center;
}}
.tile-inner {{ max-width: 1080px; margin: 0 auto; }}
.tile-light     {{ background: var(--canvas);   color: var(--ink); }}
.tile-parchment {{ background: var(--parchment); color: var(--ink); }}
.tile-dark      {{ background: var(--tile1);    color: var(--on-dark); }}
.tile-dark-2    {{ background: var(--tile2);    color: var(--on-dark); }}
.tile-dark-3    {{ background: var(--tile3);    color: var(--on-dark); }}

/* ── Pills (CTAs) ─────────────────────────────────────────── */
.pill {{
  display: inline-block;
  background: var(--primary);
  color: #ffffff !important;
  text-decoration: none;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.022em;
  padding: 11px 22px;
  border-radius: 9999px;
  border: none;
  margin: 4px;
  transition: transform 120ms ease;
}}
.pill:hover {{ filter: brightness(1.04); }}
.pill:active {{ transform: scale(0.95); }}

.pill-ghost {{
  display: inline-block;
  background: transparent;
  color: var(--primary) !important;
  border: 1px solid var(--primary);
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.022em;
  padding: 10px 22px;
  border-radius: 9999px;
  text-decoration: none;
  margin: 4px;
  transition: transform 120ms ease;
}}
.pill-ghost:active {{ transform: scale(0.95); }}

/* Streamlit button → primary pill */
.stButton > button {{
  background: var(--primary);
  color: #ffffff;
  border: none;
  border-radius: 9999px;
  padding: 11px 22px;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.022em;
  transition: transform 120ms ease;
}}
.stButton > button:hover {{ filter: brightness(1.04); color: #ffffff; }}
.stButton > button:active {{ transform: scale(0.95); }}
.stButton > button:focus {{ outline: 2px solid var(--primary-focus); outline-offset: 2px; box-shadow: none; }}

/* ── Cards ────────────────────────────────────────────────── */
.card {{
  background: var(--canvas);
  border: 1px solid var(--hairline);
  border-radius: 18px;
  padding: 24px;
  text-align: left;
}}
.card .metric-num {{
  font-family: "SF Pro Display", -apple-system, sans-serif;
  font-size: 40px;
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--ink);
}}
.card .metric-label {{
  font-size: 14px;
  color: var(--ink-48);
  letter-spacing: -0.016em;
  margin-top: 8px;
}}
.card .metric-delta-up   {{ font-size: 14px; color: var(--primary); margin-top: 4px; }}
.card .metric-delta-down {{ font-size: 14px; color: var(--ink-48); margin-top: 4px; }}

.card-dark {{
  background: var(--tile2);
  border: 1px solid rgba(255,255,255,0.06);
  color: var(--on-dark);
}}
.card-dark .metric-label {{ color: var(--muted-dark); }}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background: var(--canvas);
  border-right: 1px solid var(--hairline);
}}
section[data-testid="stSidebar"] .block-container {{ padding: 28px 20px !important; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {{
  color: var(--ink);
  font-family: "SF Pro Display", -apple-system, sans-serif;
  letter-spacing: -0.016em;
}}
section[data-testid="stSidebar"] .stRadio label {{ font-size: 14px; }}

/* Selectbox & inputs */
div[data-baseweb="select"] > div {{
  border-radius: 9999px !important;
  border-color: var(--hairline) !important;
  background: var(--canvas) !important;
}}

/* DataFrame */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--hairline);
  border-radius: 18px;
  overflow: hidden;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--hairline); }}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  border-radius: 9999px;
  padding: 8px 18px;
  font-size: 14px;
  color: var(--ink-80);
  letter-spacing: -0.016em;
}}
.stTabs [aria-selected="true"] {{
  background: var(--ink);
  color: var(--on-dark) !important;
}}

/* Streamlit alerts → quiet hairlines */
div[data-testid="stAlert"] {{
  border-radius: 18px;
  border: 1px solid var(--hairline);
  background: var(--canvas);
}}

/* Product shadow utility (only for hero imagery) */
.product-shadow {{ box-shadow: var(--product-shadow); }}

/* Plotly background */
.js-plotly-plot {{ background: transparent !important; }}

/* Footer */
.footer {{
  background: var(--parchment);
  padding: 64px 32px;
  color: var(--ink-80);
  border-top: 1px solid var(--divider-soft);
}}
.footer .cols {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 32px; max-width: 1080px; margin: 0 auto; }}
.footer h5 {{ font-size: 14px; font-weight: 600; margin: 0 0 12px 0; letter-spacing: -0.016em; }}
.footer a  {{ font-size: 12px; color: var(--ink-80); text-decoration: none; line-height: 2.4; display:block; letter-spacing: -0.01em; }}
.footer a:hover {{ color: var(--primary); }}
.footer .legal {{ max-width: 1080px; margin: 32px auto 0 auto; padding-top: 24px; border-top: 1px solid var(--divider-soft); font-size: 12px; color: var(--ink-48); }}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Inline SVG icons (no emojis)
# ─────────────────────────────────────────────────────────────
def icon(name: str, size: int = 20, color: str | None = None, accent: str | None = None) -> str:
    c = color or TOK["ink"]
    a = accent or TOK["primary"]
    icons = {
        "chart": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 3v18h18" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
            <path d="M7 14v4M12 9v9M17 12v6" stroke="{a}" stroke-width="2" stroke-linecap="round"/>
        </svg>""",
        "spark": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 16l5-7 4 4 5-9 4 6" stroke="{a}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""",
        "scatter": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 21h18" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
            <circle cx="6" cy="17" r="1.6" fill="{a}"/><circle cx="10" cy="13" r="1.6" fill="{a}"/>
            <circle cx="14" cy="15" r="1.6" fill="{a}"/><circle cx="18" cy="9"  r="1.6" fill="{a}"/>
            <circle cx="20" cy="5"  r="1.6" fill="{c}"/><circle cx="8"  cy="9" r="1.6" fill="{c}"/>
        </svg>""",
        "model": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="5"  cy="6"  r="2" stroke="{c}" stroke-width="1.4"/>
            <circle cx="5"  cy="18" r="2" stroke="{c}" stroke-width="1.4"/>
            <circle cx="12" cy="12" r="2.2" fill="{a}"/>
            <circle cx="19" cy="6"  r="2" stroke="{c}" stroke-width="1.4"/>
            <circle cx="19" cy="18" r="2" stroke="{c}" stroke-width="1.4"/>
            <path d="M7 6l3 5M7 18l3-5M14 11l3-5M14 13l3 5" stroke="{c}" stroke-width="1.2" stroke-linecap="round"/>
        </svg>""",
        "shap": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 6h10" stroke="{a}" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M4 12h14" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>
            <path d="M4 18h6" stroke="{a}" stroke-width="2.2" stroke-linecap="round"/>
        </svg>""",
        "interval": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 12h14" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
            <path d="M5 8v8M19 8v8" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
            <circle cx="12" cy="12" r="2.4" fill="{a}"/>
        </svg>""",
        "boot": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
            <circle cx="12" cy="12" r="3" fill="{a}"/>
        </svg>""",
        "cv": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="6"  width="18" height="3" rx="1.5" fill="{a}"/>
            <rect x="3" y="11" width="14" height="3" rx="1.5" fill="{c}" opacity="0.85"/>
            <rect x="3" y="16" width="10" height="3" rx="1.5" fill="{c}" opacity="0.55"/>
        </svg>""",
        "diag": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="9" stroke="{c}" stroke-width="1.4"/>
            <path d="M12 7v5l3 3" stroke="{a}" stroke-width="1.8" stroke-linecap="round"/>
        </svg>""",
        "arrow": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 12h14M13 6l6 6-6 6" stroke="{a}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""",
        "search": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="11" cy="11" r="6" stroke="{c}" stroke-width="1.4"/>
            <path d="M16 16l4 4" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
        </svg>""",
        "bag": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 8h14l-1.2 12.1a2 2 0 0 1-2 1.9H8.2a2 2 0 0 1-2-1.9L5 8z" stroke="{c}" stroke-width="1.4" stroke-linejoin="round"/>
            <path d="M9 8V6a3 3 0 1 1 6 0v2" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>
        </svg>""",
        "wave": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 14c2-3 4-3 6 0s4 3 6 0 4-3 6 0" stroke="{a}" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M3 18c2-3 4-3 6 0s4 3 6 0 4-3 6 0" stroke="{c}" stroke-width="1.4" stroke-linecap="round" opacity="0.5"/>
        </svg>""",
    }
    return icons.get(name, f'<svg width="{size}" height="{size}"><circle cx="{size//2}" cy="{size//2}" r="{size//2-2}" fill="{c}"/></svg>')


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pkl_model():
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        return None, None, None, None, {}
    with open(pkl_files[0], "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        meta = {k: obj[k] for k in (
            "best_alpha", "test_rmse", "test_mae", "test_r2",
            "nested_cv_r2_mean", "nested_cv_r2_std",
        ) if k in obj}
        return obj.get("model"), obj.get("scaler"), obj.get("features"), os.path.basename(pkl_files[0]), meta
    return obj, None, None, os.path.basename(pkl_files[0]), {}


@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "users.csv")
    if not os.path.exists(csv_path):
        # synthesize demo data so the UI still renders
        rng = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
        n = len(rng)
        np.random.seed(7)
        season_amp = 8000 * np.sin(2 * np.pi * rng.month / 12) ** 2
        base = 12000 + np.linspace(0, 4000, n)
        noise = np.random.normal(0, 1200, n)
        demo = pd.DataFrame({
            "공원명": np.random.choice(["여의도", "반포", "뚝섬", "잠원", "망원"], n),
            "현황 일시": rng,
            "일반이용자(아침)": np.clip(base * 0.25 + season_amp * 0.2 + noise * 0.4, 0, None).astype(int),
            "일반이용자(낮)":   np.clip(base * 0.45 + season_amp * 0.5 + noise * 0.5, 0, None).astype(int),
            "일반이용자(저녁)": np.clip(base * 0.30 + season_amp * 0.3 + noise * 0.4, 0, None).astype(int),
            "강수량":   np.clip(np.random.gamma(2, 35, n), 0, None).round(1),
            "평균기온": (np.sin(2 * np.pi * rng.month / 12) * 14 + 12 + np.random.normal(0, 1.5, n)).round(1),
            "미세먼지": np.clip(np.random.normal(45, 18, n), 5, None).round(1),
            "한강공원": np.clip(np.random.normal(58, 18, n) + season_amp / 600, 10, 100).round(0),
        })
        df = demo
    else:
        df = pd.read_csv(csv_path, encoding="utf-8")
        df["현황 일시"] = pd.to_datetime(df["현황 일시"])

    df["연월"] = df["현황 일시"].dt.to_period("M")
    time_cols = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]

    # 피처에서 제외할 메타/ID/주소/일시 컬럼 (이름 기반 — pandas 3.0 string dtype 대응)
    META_HINTS = ["일련번호", "코드", "주소", "시명", "구명", "등록", "수정", "일시"]
    LEAK = set(time_cols) | {"총이용객", "월"}

    def is_feature(col: str) -> bool:
        if col in LEAK or col in ("공원명", "현황 일시", "연월", "계절", "검색량"):
            return False
        return not any(h in col for h in META_HINTS)

    feature_cols = [c for c in df.columns if is_feature(c)]
    # 시간대(누수) + 피처 컬럼을 숫자로 강제 변환
    for c in time_cols + feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[time_cols].sum(axis=1)

    # 월별 집계: 시설/이용객은 모두 합계 (pkl 학습 파이프라인과 동일)
    agg_cols = ["총이용객"] + [c for c in time_cols if c in df.columns] + feature_cols
    agg_cols = list(dict.fromkeys(agg_cols))   # dedup, 순서 유지
    monthly = df.groupby("연월")[agg_cols].sum().reset_index()
    monthly["연월"] = monthly["연월"].dt.to_timestamp()
    monthly["월"] = monthly["연월"].dt.month

    def season(m: int) -> str:
        return "봄" if m in (3, 4, 5) else "여름" if m in (6, 7, 8) else "가을" if m in (9, 10, 11) else "겨울"

    monthly["계절"] = monthly["월"].apply(season)

    # 네이버 검색 트렌드 병합 → 검색량 (app.py와 동일 소스)
    trend_path = os.path.join(data_dir, "trend.xlsx")
    if os.path.exists(trend_path) and "한강공원" not in monthly.columns:
        trend = pd.read_excel(trend_path).rename(columns={"날짜": "연월"})
        trend["연월"] = pd.to_datetime(trend["연월"])
        if "한강공원" in trend.columns:
            tsub = trend[["연월", "한강공원"]].rename(columns={"한강공원": "검색량"})
            monthly = pd.merge(monthly, tsub, on="연월", how="left")
            monthly["검색량"] = monthly["검색량"].fillna(monthly["검색량"].median())
    if "검색량" not in monthly.columns:
        monthly["검색량"] = pd.to_numeric(monthly.get("한강공원", 0), errors="coerce").fillna(0)

    # 모델링 피처 = 시설 피처 + 검색량 (ID/주소/시간대 누수 제외)
    feat_final = [c for c in feature_cols if c in monthly.columns] + ["검색량"]
    feat_final = list(dict.fromkeys(feat_final))

    park_list = df["공원명"].dropna().unique().tolist() if "공원명" in df.columns else []
    return monthly, df, feat_final, time_cols, park_list


pkl_model, pkl_scaler, pkl_features, pkl_name, pkl_meta = load_pkl_model()
monthly, raw_df, num_cols, time_cols, park_list = load_data()


# ─────────────────────────────────────────────────────────────
# 모델 입력 행렬 — pkl이 아는 피처만 사용 (없으면 정제된 전체 피처)
# 누수(시간대)·ID 컬럼은 load_data 단계에서 이미 제외됨
# ─────────────────────────────────────────────────────────────
def get_Xy(df_monthly):
    if pkl_features:
        cols = [c for c in pkl_features if c in df_monthly.columns]
    else:
        cols = [c for c in num_cols if c in df_monthly.columns]
    X = df_monthly[cols].astype(float).fillna(0)
    y = df_monthly["총이용객"].values
    return X, y, cols


# ─────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, *, dark: bool = False) -> go.Figure:
    bg = TOK["tile1"] if dark else TOK["canvas"]
    ink = TOK["on_dark"] if dark else TOK["ink"]
    grid = "rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.06)"
    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(family='SF Pro Text, -apple-system, Inter, sans-serif', color=ink, size=13),
        margin=dict(l=20, r=20, t=40, b=30),
        title=dict(font=dict(family='SF Pro Display, -apple-system, sans-serif', size=20, color=ink), x=0.02),
        colorway=PLOT_PALETTE,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=dict(color=ink)),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=dict(color=ink)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────
def render_global_nav() -> None:
    st.markdown(f"""
    <div class="global-nav">
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#data">데이터</a>
        <a href="#model">모델</a>
        <a href="#interpret">해석</a>
        <a href="#diagnostics">진단</a>
        <a href="#uncertainty">불확실성</a>
      </div>
      <div class="nav-links">
        <a href="#search">{icon('search', 16, TOK['on_dark'])}</a>
        <a href="#bag">{icon('bag', 16, TOK['on_dark'])}</a>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_sub_nav(category: str) -> None:
    st.markdown(f"""
    <div class="sub-nav">
      <div class="label">{category}</div>
      <div class="right">
        <a href="#overview">개요</a>
        <a href="#features">기능</a>
        <a href="#model">모델</a>
        <a href="#report">리포트</a>
        <a class="pill" href="#start">시작하기 {icon('arrow', 14, '#fff', '#fff')}</a>
      </div>
    </div>
    """, unsafe_allow_html=True)


def tile_open(kind: str = "light", anchor: str | None = None) -> None:
    cls = {
        "light":     "tile tile-light",
        "parchment": "tile tile-parchment",
        "dark":      "tile tile-dark",
        "dark2":     "tile tile-dark-2",
        "dark3":     "tile tile-dark-3",
    }.get(kind, "tile tile-light")
    a = f' id="{anchor}"' if anchor else ""
    st.markdown(f'<section{a} class="{cls}"><div class="tile-inner">', unsafe_allow_html=True)


def tile_close() -> None:
    st.markdown('</div></section>', unsafe_allow_html=True)


# 공원 좌표 (사이드바 미니 지도 + 개요 큰 지도 공용)
PARK_COORDS = {
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


def nearest_park(lat: float, lng: float) -> str:
    import math
    return min(PARK_COORDS.items(),
               key=lambda kv: math.hypot(lat - kv[1][0], lng - kv[1][1]))[0]


def metric_card(num: str, label: str, delta: str | None = None, *, dark: bool = False) -> str:
    klass = "card card-dark" if dark else "card"
    delta_html = ""
    if delta:
        cls = "metric-delta-up" if delta.startswith("+") else "metric-delta-down"
        delta_html = f'<div class="{cls}">{delta}</div>'
    return f"""
    <div class="{klass}">
      <div class="metric-num">{num}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>
    """


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
      <div style="width:36px;height:36px;border-radius:9999px;background:{TOK['ink']};
                  display:flex;align-items:center;justify-content:center;">
        {icon('wave', 20, TOK['on_dark'], TOK['primary_on_dark'])}
      </div>
      <div>
        <div class="body-strong" style="margin:0">한강 Analytics</div>
        <div class="caption" style="margin:0">v2.0 · Retheme</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 지도/네비 클릭 결과를 위젯 생성 '전'에 반영 (위젯 키는 생성 후 수정 불가)
    if "_map_pick" in st.session_state:
        st.session_state["selected_park"] = st.session_state.pop("_map_pick")
    if "_nav" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_nav")

    if park_list:
        st.selectbox("공원 선택", park_list, key="selected_park")
    else:
        st.text_input("공원명", value="여의도", key="selected_park")
    selected_park = st.session_state.get("selected_park")

    # 미니 지도 — 모든 페이지에서 공원 선택 가능
    _mini = folium.Map(location=[37.53, 126.98], zoom_start=10,
                       tiles="CartoDB positron", zoom_control=False,
                       scrollWheelZoom=False, dragging=False)
    for _n, _c in PARK_COORDS.items():
        _s = (_n == selected_park)
        folium.CircleMarker(
            location=_c, radius=7 if _s else 4,
            color="#E8505B" if _s else "#0066cc",
            weight=2, fill=True,
            fill_color="#E8505B" if _s else "#0066cc", fill_opacity=0.95,
            tooltip=_n, popup=_n,
        ).add_to(_mini)
    _md = st_folium(_mini, width=280, height=200, key="mini_map")
    if _md:
        _o = _md.get("last_object_clicked") or _md.get("last_clicked")
        if _o:
            _np = nearest_park(_o["lat"], _o["lng"])
            if _np != selected_park and _np in (park_list or [_np]):
                st.session_state["_map_pick"] = _np
                st.rerun()
    st.markdown(f'<div class="caption" style="margin:-6px 0 10px 0">선택: '
                f'<b style="color:var(--primary)">{selected_park or "-"}</b></div>',
                unsafe_allow_html=True)

    PAGES = [
        ("개요",            "spark"),
        ("EDA",            "chart"),
        ("t-test & VIF",   "scatter"),
        ("모델 예측",       "model"),
        ("예측 시뮬레이터",  "boot"),
        ("잔차 진단",       "diag"),
        ("SHAP 해석",       "shap"),
        ("Conformal",      "interval"),
        ("Bootstrap CI",   "boot"),
        ("Nested CV",      "cv"),
    ]
    page = st.radio("분석", [p[0] for p in PAGES], key="page", label_visibility="visible")

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="caption" style="line-height:1.7">
      <div class="body-strong" style="color:var(--ink); margin-bottom:6px">데이터 요약</div>
      관측 월수 · {len(monthly):,}<br/>
      변수 수 · {len(num_cols)}<br/>
      모델 · {pkl_name or '없음'}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Global + sub nav
# ─────────────────────────────────────────────────────────────
render_global_nav()
render_sub_nav(f"한강공원 · {selected_park}" if selected_park else "한강공원")


# ─────────────────────────────────────────────────────────────
# 지도 — 페이지 최상단 (히어로 바로 위). 개요 페이지에서만 표시
# ─────────────────────────────────────────────────────────────
if page == "개요":
    _m = folium.Map(location=[37.53, 126.98], zoom_start=12, tiles="CartoDB positron")
    for _name, _coord in PARK_COORDS.items():
        _sel = (_name == selected_park)
        folium.Marker(
            location=_coord, tooltip=_name, popup=_name,
            icon=folium.Icon(color="red" if _sel else "blue",
                             icon="star" if _sel else "info-sign"),
        ).add_to(_m)

    tile_open("light")
    st.markdown('<h2 class="h-section" style="text-align:center; margin-bottom:16px">'
                '지도에서 공원을 선택하세요</h2>', unsafe_allow_html=True)
    _map_data = st_folium(_m, width=1100, height=460, key="overview_map")
    tile_close()

    # 마커 클릭 → 가장 가까운 공원 선택 → 다음 run 에서 사이드바에 반영
    _click = None
    if _map_data:
        _obj = _map_data.get("last_object_clicked") or _map_data.get("last_clicked")
        if _obj:
            _click = (_obj["lat"], _obj["lng"])
    if _click is not None:
        _near = nearest_park(_click[0], _click[1])
        if _near != selected_park and _near in (park_list or [_near]):
            st.session_state["_map_pick"] = _near
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Hero tile — light
# ─────────────────────────────────────────────────────────────
tile_open("light", anchor="overview")
st.markdown("""
<h1 class="h-hero">한강공원 이용객을, 데이터로.</h1>
<p class="lead">EDA부터 SHAP, Conformal, Bootstrap, Nested CV까지 — 하나의 워크플로우.</p>
""", unsafe_allow_html=True)

# 히어로 CTA — 실제 페이지 이동
if page == "개요":
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    _b = st.columns([3, 1.3, 1.3, 3])
    with _b[1]:
        if st.button("모델 실행하기", key="hero_to_sim", use_container_width=True):
            st.session_state["_nav"] = "예측 시뮬레이터"
            st.rerun()
    with _b[2]:
        if st.button("기능 살펴보기", key="hero_to_eda", use_container_width=True):
            st.session_state["_nav"] = "EDA"
            st.rerun()

# Top metric row
if len(monthly) > 0:
    total_users   = int(monthly["총이용객"].sum())
    avg_monthly   = int(monthly["총이용객"].mean())
    peak_idx      = monthly["총이용객"].idxmax()
    peak_month    = monthly.loc[peak_idx, "연월"].strftime("%Y년 %m월")
    trend_pct     = (monthly["총이용객"].iloc[-12:].mean()
                     / max(monthly["총이용객"].iloc[:12].mean(), 1) - 1) * 100

    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: st.markdown(metric_card(f"{total_users/1e6:.1f}M", "누적 이용객"),                       unsafe_allow_html=True)
    with c2: st.markdown(metric_card(f"{avg_monthly/1000:.0f}K", "월 평균 이용객"),                  unsafe_allow_html=True)
    with c3: st.markdown(metric_card(peak_month,                "최대 방문 월"),                      unsafe_allow_html=True)
    with c4: st.markdown(metric_card(f"{trend_pct:+.1f}%",       "최근 1년 추세",
                                     delta=("+상승세" if trend_pct >= 0 else "감소세")),             unsafe_allow_html=True)

tile_close()


# ─────────────────────────────────────────────────────────────
# Page routing — alternating tiles
# ─────────────────────────────────────────────────────────────

if page == "개요":
    # Dark tile — features grid
    tile_open("dark", anchor="features")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--on-dark)">하나의 대시보드, 여덟 가지 분석.</h2>
    <p class="lead lead-on-dark">통계 검정과 머신러닝, 해석가능성과 불확실성을 한 화면에서.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:48px"></div>', unsafe_allow_html=True)

    rows = [
        [("chart",    "EDA",            "월별 추이 · 계절성 · 변수 분포를 한눈에."),
         ("scatter",  "t-test & VIF",   "유의미한 피처를 골라내고, 공선성을 제거."),
         ("model",    "예측 모델",      "사전학습 pkl 모델로 즉시 추론.")],
        [("diag",     "잔차 진단",      "Q-Q, 등분산성, 자기상관까지 정량 평가."),
         ("shap",     "SHAP",           "전역·국소 기여도로 의사결정을 설명."),
         ("interval", "Conformal",      "분포 가정 없이 예측 구간을 보장.")],
    ]
    for row in rows:
        cols = st.columns(3, gap="medium")
        for col, (ic, title, desc) in zip(cols, row):
            col.markdown(f"""
            <div class="card card-dark" style="min-height: 180px;">
              <div style="margin-bottom:14px">{icon(ic, 24, TOK['on_dark'], TOK['primary_on_dark'])}</div>
              <div class="body-strong" style="color:var(--on-dark)">{title}</div>
              <div class="caption" style="color:var(--muted-dark); margin-top:6px; line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    tile_close()

    # 지도는 페이지 최상단(히어로 위)으로 이동됨. 여기서는 데이터 검증만.
    with st.expander("데이터 요약 및 검증"):
        try:
            st.write(f"원본 행 수: {len(raw_df)}")
            st.write(f"월별 집계 행 수: {monthly.shape[0]}")
            st.write(f"월별 총이용객 합계 (monthly): {int(monthly['총이용객'].sum())}")
            if '총이용객' in raw_df.columns:
                st.write(f"원본 총이용객 합계 (raw): {int(raw_df['총이용객'].sum())}")
                grp = raw_df.groupby('연월')['총이용객'].sum()
                st.write(f"원본으로 연월별 그룹핑 후 합계 합: {int(grp.sum())}")
                st.write(f"집계 검증 차이 (monthly.sum - raw_group_sum): {int(monthly['총이용객'].sum() - grp.sum())}")
        except Exception as e:
            st.write("데이터 요약을 계산하는 동안 오류가 발생했습니다:", e)

    # Parchment tile — 선택 공원 월별 추이
    tile_open("parchment", anchor="sample")
    st.markdown('<h2 class="h-section">월별 이용객 추이 (전체 합계)</h2>', unsafe_allow_html=True)
    fig = px.line(monthly, x="연월", y="총이용객")
    fig.update_traces(line=dict(color=TOK["primary"], width=2.4))
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tile_close()

    # Light tile — 공원별 평균 이용객 랭킹 (선택 공원 강조)
    if "공원명" in raw_df.columns and "총이용객" in raw_df.columns:
        tile_open("light")
        st.markdown('<h2 class="h-section">공원별 평균 이용객</h2>', unsafe_allow_html=True)
        rank = (raw_df.groupby("공원명")["총이용객"].mean()
                .sort_values().reset_index())
        bar_colors = [("#E8505B" if p == selected_park else TOK["primary"])
                      for p in rank["공원명"]]
        figr = go.Figure(go.Bar(
            x=rank["총이용객"], y=rank["공원명"], orientation="h",
            marker_color=bar_colors,
            text=[f"{v/1e4:.0f}만" for v in rank["총이용객"]], textposition="outside"))
        figr.update_layout(height=460,
                           title=f"빨강 = 선택 공원 ({selected_park})",
                           xaxis_title="월평균 이용객 수")
        style_fig(figr)
        st.plotly_chart(figr, use_container_width=True, config={"displayModeBar": False})
        tile_close()


elif page == "EDA":
    tile_open("light", anchor="eda")
    st.markdown('<h2 class="h-section">탐색적 데이터 분석</h2>', unsafe_allow_html=True)

    tabs = st.tabs(["월별 추이", "시간대 구성", "계절성", "변수 분포"])

    with tabs[0]:
        fig = px.line(monthly, x="연월", y="총이용객")
        fig.update_traces(line=dict(color=TOK["primary"], width=2.4))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tabs[1]:
        df_long = monthly.melt(id_vars="연월", value_vars=time_cols, var_name="시간대", value_name="이용객")
        fig = px.area(df_long, x="연월", y="이용객", color="시간대",
                      color_discrete_sequence=[TOK["primary"], TOK["ink"], TOK["primary_on_dark"]])
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tabs[2]:
        season_df = monthly.groupby("계절")["총이용객"].mean().reindex(["봄", "여름", "가을", "겨울"]).reset_index()
        fig = px.bar(season_df, x="계절", y="총이용객")
        fig.update_traces(marker_color=TOK["primary"], marker_line_width=0)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tabs[3]:
        choices = [c for c in num_cols if c in monthly.columns][:6]
        if choices:
            picked = st.selectbox("변수", choices)
            fig = px.histogram(monthly, x=picked, nbins=24)
            fig.update_traces(marker_color=TOK["primary"], marker_line_width=0)
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tile_close()


elif page == "t-test & VIF":
    tile_open("parchment", anchor="ttest")
    st.markdown('<h2 class="h-section">t-test 기반 피처 선택 · VIF</h2>', unsafe_allow_html=True)

    try:
        from scipy import stats
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        y = monthly["총이용객"].values
        thresh = np.median(y)
        groups = (y >= thresh).astype(int)

        feats = [c for c in num_cols if c in monthly.columns and c not in ("총이용객", "월")]
        rows = []
        for f in feats:
            x = monthly[f].values
            t, p = stats.ttest_ind(x[groups == 1], x[groups == 0], equal_var=False)
            rows.append({"변수": f, "t-stat": round(t, 3), "p-value": round(p, 4),
                         "유의성": "유의" if p < 0.05 else "—"})
        ttest_df = pd.DataFrame(rows).sort_values("p-value")

        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown('<div class="body-strong" style="margin-bottom:10px">t-test 결과</div>', unsafe_allow_html=True)
            st.dataframe(ttest_df, use_container_width=True, hide_index=True)
        with c2:
            sig = ttest_df[ttest_df["유의성"] == "유의"]["변수"].tolist()
            if len(sig) >= 2:
                X = monthly[sig].astype(float).fillna(0)
                vif = pd.DataFrame({
                    "변수": sig,
                    "VIF": [round(variance_inflation_factor(X.values, i), 2) for i in range(X.shape[1])],
                })
                st.markdown('<div class="body-strong" style="margin-bottom:10px">VIF (공선성)</div>', unsafe_allow_html=True)
                st.dataframe(vif, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="caption">유의 피처가 부족합니다.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="caption">분석을 실행할 수 없습니다 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "모델 예측":
    tile_open("dark", anchor="model")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--on-dark)">사전학습 모델로, 한 번에 추론.</h2>
    <p class="lead lead-on-dark">model/ 폴더의 pkl을 로드해 예측 · 잔차 · 지표를 함께 산출합니다.</p>
    """, unsafe_allow_html=True)
    tile_close()

    tile_open("light")
    st.markdown('<h2 class="h-section">모델 예측 결과</h2>', unsafe_allow_html=True)

    if pkl_model is None:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:48px 24px;">
          <div style="margin-bottom:16px">{icon('model', 32, TOK['ink_48'])}</div>
          <div class="body-strong">모델 파일을 찾을 수 없습니다</div>
          <div class="caption" style="margin-top:6px">algo-harness/model/ 폴더에 .pkl 파일을 두세요.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score, mean_absolute_error
            from sklearn.preprocessing import StandardScaler

            X, y, used_cols = get_Xy(monthly)   # pkl이 아는 8개 피처 (검색량 포함)

            if pkl_scaler is not None and pkl_features and len(used_cols) == len(pkl_features):
                preds = pkl_model.predict(pkl_scaler.transform(X))
                used_pkl = True
            else:
                # pkl을 쓸 수 없을 때만 동일 피처로 Ridge 재학습 (누수 컬럼 없음)
                scaler = StandardScaler().fit(X)
                preds = Ridge(alpha=1.0).fit(scaler.transform(X), y).predict(scaler.transform(X))
                used_pkl = False

            # 현재 데이터 재계산 지표 (학습 데이터 포함 → 낙관적일 수 있음)
            r2  = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)

            # 표시 지표: pkl 저장된 검증값 우선
            disp_r2  = pkl_meta.get("test_r2",  r2)  if used_pkl else r2
            disp_mae = pkl_meta.get("test_mae", mae) if used_pkl else mae

            c1, c2, c3 = st.columns(3, gap="medium")
            c1.markdown(metric_card(f"{disp_r2:.3f}",        "R²"),          unsafe_allow_html=True)
            c2.markdown(metric_card(f"{disp_mae/1000:.1f}K", "MAE"),         unsafe_allow_html=True)
            c3.markdown(metric_card(pkl_name or "Ridge",     "사용 모델"),   unsafe_allow_html=True)

            if used_pkl and pkl_meta:
                st.markdown(
                    f'<div class="caption" style="margin-top:12px">검증셋 기준 저장값(pkl). '
                    f'현재 데이터 재예측: R² {r2:.3f} · MAE {mae/1000:.1f}K.</div>',
                    unsafe_allow_html=True)
            elif not used_pkl:
                st.markdown(
                    '<div class="caption" style="margin-top:12px">⚠️ pkl 피처와 불일치하여 동일 피처로 Ridge를 재학습했습니다.</div>',
                    unsafe_allow_html=True)

            st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["연월"], y=y,     name="실측",
                                     line=dict(color=TOK["ink"], width=1.8)))
            fig.add_trace(go.Scatter(x=monthly["연월"], y=preds, name="예측",
                                     line=dict(color=TOK["primary"], width=2.4)))
            fig.update_layout(title="실측 vs 예측")
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.markdown(f'<div class="caption">예측 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "예측 시뮬레이터":
    tile_open("dark", anchor="simulator")
    st.markdown("""
    <h2 class="h-display" style="color:var(--on-dark)">입력을 바꾸면, 예측이 즉시.</h2>
    <p class="lead lead-on-dark">시설 규모와 검색량을 조정하면 사전학습 모델이 월 이용객을 다시 추정합니다.</p>
    """, unsafe_allow_html=True)
    tile_close()

    tile_open("light")
    if pkl_model is None or not pkl_features:
        st.markdown('<div class="card">모델(pkl) 또는 피처 정보가 없어 시뮬레이터를 사용할 수 없습니다.</div>',
                    unsafe_allow_html=True)
    else:
        X_sim, y_sim, sim_cols = get_Xy(monthly)

        st.markdown('<h2 class="h-section">입력 값 조정</h2>', unsafe_allow_html=True)
        st.markdown('<div class="caption" style="margin-bottom:18px">'
                    '기본값은 각 변수의 월별 중앙값입니다. 슬라이더를 움직이면 예측이 즉시 갱신됩니다.</div>',
                    unsafe_allow_html=True)

        scols = st.columns(2, gap="large")
        vals = {}
        for i, c in enumerate(sim_cols):
            s = X_sim[c]
            lo, hi, med = float(s.min()), float(s.max()), float(s.median())
            if hi <= lo:
                hi = lo + 1.0
            step = max((hi - lo) / 100.0, 1.0)
            with scols[i % 2]:
                vals[c] = st.slider(c, lo, hi, med, step=step)

        Xrow = pd.DataFrame([[vals[c] for c in sim_cols]], columns=sim_cols)
        try:
            pred = float(pkl_model.predict(pkl_scaler.transform(Xrow))[0])
        except Exception as e:
            pred = None
            st.markdown(f'<div class="caption">예측 오류 — {e}</div>', unsafe_allow_html=True)

        if pred is not None:
            avg = float(np.mean(y_sim))
            diff = (pred - avg) / avg * 100 if avg else 0.0

            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2, gap="medium")
            r1.markdown(metric_card(f"{pred/1e4:,.1f}만 명", "예측 월 이용객",
                                    delta=f"{'+' if diff >= 0 else ''}{diff:.1f}% vs 평균"),
                        unsafe_allow_html=True)
            r2.markdown(metric_card(f"{avg/1e4:,.1f}만 명", "전체 월평균 (실측)"),
                        unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=["예측값", "전체 평균"], y=[pred, avg],
                marker_color=[TOK["primary"], TOK["ink_48"]],
                text=[f"{pred/1e4:.1f}만", f"{avg/1e4:.1f}만"], textposition="outside",
                width=[0.5, 0.5]))
            fig.update_layout(title="예측 vs 실측 월평균", height=360, yaxis_title="월 이용객 수")
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tile_close()


elif page == "잔차 진단":
    tile_open("light", anchor="diagnostics")
    st.markdown('<h2 class="h-section">잔차 도표 · 진단</h2>', unsafe_allow_html=True)

    try:
        from scipy.stats import probplot
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y, _ = get_Xy(monthly)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
        preds = model.predict(scaler.transform(X))
        resid = y - preds

        c1, c2 = st.columns(2, gap="medium")

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=preds, y=resid, mode="markers",
                                     marker=dict(color=TOK["primary"], size=7, opacity=0.85)))
            fig.add_hline(y=0, line=dict(color=TOK["ink"], width=1, dash="dot"))
            fig.update_layout(title="잔차 vs 예측값", xaxis_title="예측", yaxis_title="잔차")
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            (osm, osr), _ = probplot(resid, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(color=TOK["primary"], size=7)))
            fig.add_trace(go.Scatter(x=osm, y=osm * np.std(resid),
                                     mode="lines", line=dict(color=TOK["ink"], dash="dot")))
            fig.update_layout(title="Q-Q Plot", showlegend=False)
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">진단 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "SHAP 해석":
    tile_open("parchment", anchor="interpret")
    st.markdown('<h2 class="h-section">SHAP 기반 해석</h2>', unsafe_allow_html=True)

    try:
        import shap
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y, _ = get_Xy(monthly)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = Ridge(alpha=1.0).fit(Xs, y)

        explainer = shap.LinearExplainer(model, Xs)
        sv = explainer.shap_values(Xs)
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({"변수": X.columns, "기여도": mean_abs}).sort_values("기여도", ascending=True)

        fig = px.bar(imp, x="기여도", y="변수", orientation="h")
        fig.update_traces(marker_color=TOK["primary"], marker_line_width=0)
        fig.update_layout(title="전역 기여도 (평균 |SHAP|)", height=420)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">SHAP 분석 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "Conformal":
    tile_open("dark2", anchor="uncertainty")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--on-dark)">분포 가정 없는 예측 구간.</h2>
    <p class="lead lead-on-dark">Split Conformal로 보장된 커버리지를 — 단 하나의 캘리브레이션 단계로.</p>
    """, unsafe_allow_html=True)
    tile_close()

    tile_open("light")
    st.markdown('<h2 class="h-section">Conformal Prediction</h2>', unsafe_allow_html=True)

    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        alpha = st.slider("유의수준 α", 0.05, 0.30, 0.10, 0.05,
                          help="1-α 가 커버리지 (예: α=0.10 → 90% 구간)")

        X, y, _ = get_Xy(monthly)
        idx = np.arange(len(y))
        X_tr, X_cal, y_tr, y_cal, i_tr, i_cal = train_test_split(X, y, idx, test_size=0.3, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        model = Ridge(alpha=1.0).fit(scaler.transform(X_tr), y_tr)
        cal_preds = model.predict(scaler.transform(X_cal))
        residuals = np.abs(y_cal - cal_preds)
        q = np.quantile(residuals, 1 - alpha)
        preds = model.predict(scaler.transform(X))
        lower = preds - q
        upper = preds + q
        coverage = float(np.mean((y >= lower) & (y <= upper))) * 100

        c1, c2 = st.columns([1, 3], gap="medium")
        c1.markdown(metric_card(f"{coverage:.1f}%", "실측 커버리지", delta=f"+목표 {(1-alpha)*100:.0f}%"),
                    unsafe_allow_html=True)

        with c2:
            order = np.argsort(monthly["연월"].values)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["연월"].iloc[order], y=upper[order],
                                     mode="lines", line=dict(color="rgba(0,102,204,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=monthly["연월"].iloc[order], y=lower[order],
                                     mode="lines", line=dict(color="rgba(0,102,204,0)"),
                                     fill="tonexty", fillcolor="rgba(0,102,204,0.18)", name=f"{int((1-alpha)*100)}% 구간"))
            fig.add_trace(go.Scatter(x=monthly["연월"].iloc[order], y=preds[order],
                                     mode="lines", name="예측", line=dict(color=TOK["primary"], width=2.4)))
            fig.add_trace(go.Scatter(x=monthly["연월"].iloc[order], y=y[order],
                                     mode="markers", name="실측",
                                     marker=dict(color=TOK["ink"], size=5)))
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">Conformal 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "Bootstrap CI":
    tile_open("parchment", anchor="bootstrap")
    st.markdown('<h2 class="h-section">Bootstrap 95% 신뢰구간</h2>', unsafe_allow_html=True)

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        n_boot = st.slider("Bootstrap 반복 수", 100, 1000, 300, 100)

        X_df, y, names = get_Xy(monthly)
        X = X_df.values
        n = len(y)
        rng = np.random.default_rng(7)
        boot_coefs = []
        for _ in range(n_boot):
            samp = rng.integers(0, n, n)
            sc = StandardScaler().fit(X[samp])
            mdl = Ridge(alpha=1.0).fit(sc.transform(X[samp]), y[samp])
            boot_coefs.append(mdl.coef_)
        boot_coefs = np.array(boot_coefs)
        lo = np.percentile(boot_coefs, 2.5, axis=0)
        hi = np.percentile(boot_coefs, 97.5, axis=0)
        mean = boot_coefs.mean(axis=0)
        ci_df = pd.DataFrame({"변수": names, "평균계수": mean, "하한": lo, "상한": hi})
        ci_df = ci_df.sort_values("평균계수")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ci_df["평균계수"], y=ci_df["변수"],
            error_x=dict(type="data", symmetric=False,
                         array=ci_df["상한"] - ci_df["평균계수"],
                         arrayminus=ci_df["평균계수"] - ci_df["하한"],
                         color=TOK["ink_48"], thickness=1.5),
            mode="markers", marker=dict(color=TOK["primary"], size=10),
            name="95% CI",
        ))
        fig.add_vline(x=0, line=dict(color=TOK["ink"], dash="dot"))
        fig.update_layout(title=f"Bootstrap 계수 분포 (n={n_boot})", height=480)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">Bootstrap 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "Nested CV":
    tile_open("light", anchor="nested-cv")
    st.markdown('<h2 class="h-section">Nested Cross-Validation</h2>', unsafe_allow_html=True)

    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        outer_k = st.slider("Outer fold", 3, 7, 5)
        inner_k = st.slider("Inner fold", 3, 5, 3)
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

        X_df, y, _ = get_Xy(monthly)
        X = X_df.values

        outer = KFold(n_splits=outer_k, shuffle=True, random_state=42)
        outer_scores, chosen = [], []
        for tr, te in outer.split(X):
            best_a, best_s = None, -np.inf
            for a in alphas:
                inner = KFold(n_splits=inner_k, shuffle=True, random_state=0)
                s = []
                for itr, iva in inner.split(X[tr]):
                    sc = StandardScaler().fit(X[tr][itr])
                    m = Ridge(alpha=a).fit(sc.transform(X[tr][itr]), y[tr][itr])
                    s.append(r2_score(y[tr][iva], m.predict(sc.transform(X[tr][iva]))))
                if np.mean(s) > best_s:
                    best_s, best_a = float(np.mean(s)), a
            chosen.append(best_a)
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=best_a).fit(sc.transform(X[tr]), y[tr])
            outer_scores.append(r2_score(y[te], m.predict(sc.transform(X[te]))))

        c1, c2, c3 = st.columns(3, gap="medium")
        c1.markdown(metric_card(f"{np.mean(outer_scores):.3f}", "평균 R² (외부)"),       unsafe_allow_html=True)
        c2.markdown(metric_card(f"±{np.std(outer_scores):.3f}", "표준편차"),              unsafe_allow_html=True)
        c3.markdown(metric_card(f"{np.median(chosen):.2f}",     "선택 α (median)"),       unsafe_allow_html=True)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        cv_df = pd.DataFrame({"Fold": list(range(1, outer_k + 1)),
                              "R²": np.round(outer_scores, 4),
                              "선택 α": chosen})
        fig = px.bar(cv_df, x="Fold", y="R²", text="선택 α")
        fig.update_traces(marker_color=TOK["primary"], marker_line_width=0,
                          textposition="outside", textfont=dict(color=TOK["ink"]))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">Nested CV 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div class="cols">
    <div>
      <h5>분석</h5>
      <a href="#eda">EDA</a>
      <a href="#ttest">t-test &amp; VIF</a>
      <a href="#model">모델 예측</a>
      <a href="#diagnostics">잔차 진단</a>
    </div>
    <div>
      <h5>해석</h5>
      <a href="#interpret">SHAP</a>
      <a href="#uncertainty">Conformal</a>
      <a href="#bootstrap">Bootstrap CI</a>
      <a href="#nested-cv">Nested CV</a>
    </div>
    <div>
      <h5>리소스</h5>
      <a href="#">데이터 소스</a>
      <a href="#">모델 카드</a>
      <a href="#">방법론</a>
      <a href="#">변경 로그</a>
    </div>
    <div>
      <h5>관리</h5>
      <a href="#">관리자</a>
      <a href="#">권한</a>
      <a href="#">로깅</a>
      <a href="#">문의</a>
    </div>
  </div>
  <div class="legal">한강공원 Analytics Dashboard · 데이터 출처 · 서울특별시 한강사업본부 · 본 페이지는 분석 데모입니다.</div>
</div>
""", unsafe_allow_html=True)
