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

# ── 한글 폰트 (matplotlib: SHAP 등 정적 플롯용)
import matplotlib.font_manager as fm
sys_name = platform.system()
if sys_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif sys_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    # Streamlit Cloud(Linux): packages.txt 의 fonts-nanum 등록
    _set = False
    for _fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/nanum/NanumGothic.ttf"):
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = "NanumGothic"
            _set = True
            break
    if not _set:
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
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@latest/dist/web/variable/pretendardvariable-dynamic-subset.css');
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
  font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
               "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.01em;
}}

.stApp {{
  background: var(--parchment);
}}

#MainMenu, footer {{
  visibility: hidden;
  height: 0;
}}
/* 헤더 바는 투명 처리하되 제거하지 않음 → 사이드바 펼침 토글이 항상 살아있음 */
header[data-testid="stHeader"] {{
  background: transparent;
}}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
  display: flex !important;
  visibility: visible !important;
}}

.block-container {{
  padding: 0 !important;
  max-width: 100% !important;
}}

/* ── Display typography ───────────────────────────────────── */
.h-hero {{
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", system-ui, sans-serif;
  font-size: 56px;
  font-weight: 600;
  line-height: 1.07;
  letter-spacing: -0.028em;
  margin: 0;
}}
.h-display {{
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", system-ui, sans-serif;
  font-size: 40px;
  font-weight: 600;
  line-height: 1.10;
  letter-spacing: -0.02em;
  margin: 0;
}}
.h-section {{
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", system-ui, sans-serif;
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
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", system-ui, sans-serif;
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
  font-family: "Pretendard Variable", Pretendard, -apple-system, "Apple SD Gothic Neo", system-ui, sans-serif;
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

    df["연월"] = df["현황 일시"].dt.to_period("M").dt.to_timestamp()
    df["월"] = df["연월"].dt.month
    time_cols = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]

    # 피처에서 제외할 메타/ID/주소/일시 컬럼 (이름 기반 — pandas 3.0 string dtype 대응)
    META_HINTS = ["일련번호", "코드", "주소", "시명", "구명", "등록", "수정", "일시"]
    EXCL = set(time_cols) | {"총이용객", "월", "공원명", "현황 일시", "연월", "계절", "검색량", "월sin", "월cos"}

    def is_feature(col: str) -> bool:
        return col not in EXCL and not any(h in col for h in META_HINTS)

    base_feats = [c for c in df.columns if is_feature(c)]
    for c in time_cols + base_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[time_cols].sum(axis=1)

    def season(m: int) -> str:
        return "봄" if m in (3, 4, 5) else "여름" if m in (6, 7, 8) else "가을" if m in (9, 10, 11) else "겨울"

    df["계절"] = df["월"].map(season)
    # 계절성(실제 달력 기반 주기 피처)
    df["월sin"] = np.sin(2 * np.pi * df["월"] / 12)
    df["월cos"] = np.cos(2 * np.pi * df["월"] / 12)

    # 네이버 트렌드: 공원별(long) → 공원-월 검색량, 통합("한강공원") → 월별 집계용
    integ = None
    trend_path = os.path.join(data_dir, "trend.xlsx")
    if os.path.exists(trend_path):
        tr = pd.read_excel(trend_path).rename(columns={"날짜": "연월"})
        tr["연월"] = pd.to_datetime(tr["연월"])
        pcols = [c for c in tr.columns if c.endswith("한강공원") and c != "한강공원"]
        if pcols:
            trl = tr.melt(id_vars="연월", value_vars=pcols, var_name="공원명", value_name="검색량")
            df = pd.merge(df, trl, on=["연월", "공원명"], how="left")
        if "한강공원" in tr.columns:
            integ = tr[["연월", "한강공원"]].rename(columns={"한강공원": "검색량"})
    if "검색량" not in df.columns:
        df["검색량"] = np.nan
    df["검색량"] = pd.to_numeric(df["검색량"], errors="coerce")
    df["검색량"] = df["검색량"].fillna(df["검색량"].median() if df["검색량"].notna().any() else 0.0)

    # 시설 피처(분산 0 = 정보 없음 제외) + 검색량 + 계절성
    fac = [c for c in base_feats if df[c].std() > 0]
    feature_cols = fac + ["검색량", "월sin", "월cos"]

    keep = ["공원명", "연월", "월", "계절", "총이용객"] + fac + ["검색량", "월sin", "월cos"] \
        + [c for c in time_cols if c in df.columns]
    pm = df[list(dict.fromkeys(keep))].copy()    # 공원-월 단위 (≈815행)

    # 월별 집계 (설명용 차트/Overview)
    monthly = pm.groupby("연월")[["총이용객"] + [c for c in time_cols if c in pm.columns] + fac].sum().reset_index()
    monthly["월"] = monthly["연월"].dt.month
    monthly["계절"] = monthly["월"].map(season)
    if integ is not None:
        monthly = pd.merge(monthly, integ, on="연월", how="left")
        monthly["검색량"] = monthly["검색량"].fillna(monthly["검색량"].median())
    else:
        monthly = pd.merge(monthly, pm.groupby("연월")["검색량"].mean().reset_index(), on="연월", how="left")

    park_list = pm["공원명"].dropna().unique().tolist()
    return monthly, pm, feature_cols, time_cols, park_list


monthly, pm, num_cols, time_cols, park_list = load_data()
raw_df = pm                      # Overview 랭킹/검증 expander 호환
feature_cols = num_cols          # 모델 피처 = 시설 + 검색량 + 계절성


# ─────────────────────────────────────────────────────────────
# 모델: 공원-월(815행) RandomForest (공원 원핫 + 공원별 검색량 + 계절성)
#   · 실데이터·실분석. 학습/검증을 인앱에서 직접 수행 (캐시)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_bundle(_pm, feats):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

    X = _pm[feats + ["공원명"]].copy()
    y = _pm["총이용객"].values
    pre = ColumnTransformer([
        ("num", "passthrough", feats),
        ("park", OneHotEncoder(handle_unknown="ignore"), ["공원명"]),
    ])
    model = Pipeline([("pre", pre),
                      ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])
    kf = KFold(5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X, y, cv=kf, scoring="r2")
    oof = cross_val_predict(model, X, y, cv=kf)        # 폴드별 검증 예측 (정직한 대표 성능)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    return {"model": model, "X": X, "y": y, "Xtr": Xtr, "Xte": Xte,
            "ytr": ytr, "yte": yte, "cv": cv, "oof": oof, "names": names}


bundle = get_bundle(pm, feature_cols)


def get_Xy(df_in=None):
    src = pm if df_in is None else df_in
    cols = [c for c in feature_cols if c in src.columns]
    return src[cols].astype(float).fillna(0), src["총이용객"].values, cols


# ─────────────────────────────────────────────────────────────
# HSKR 비교 번들 로드 (직접 구현한 Hybrid Seasonal Kernel Ridge)
#   · pkl 언피클 전 반드시 hskr_model 모듈 import 필요
#   · 파일: hskr_model.py (루트) + model/hskr_model.pkl (또는 루트)
# ─────────────────────────────────────────────────────────────
# 캐시하지 않음 — pkl 구조가 바뀌어도 stale 캐시로 KeyError 나지 않도록 매번 새로 읽음(파일 작음)
def load_hskr():
    """HSKR 번들 로드. (bundle|None, err|None) 반환 — 실패해도 앱이 죽지 않도록 방어.

    Cloud에서 numpy/sklearn 버전이 pkl 생성 환경과 다르면 pickle.load가
    ModuleNotFoundError를 던질 수 있어, import·언피클을 모두 try로 감싼다.
    """
    base = os.path.dirname(__file__)
    for p in (os.path.join(base, "model", "hskr_model.pkl"),
              os.path.join(base, "hskr_model.pkl")):
        if os.path.exists(p):
            try:
                from hskr_model import HybridSeasonalKernelRidge  # noqa: F401  (언피클에 필요)
                with open(p, "rb") as f:
                    return pickle.load(f), None
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"
    return None, "파일 없음: model/hskr_model.pkl (또는 루트)"


# ─────────────────────────────────────────────────────────────
# 피처 중요도 재학습 비교 번들 로드 (fi_models.pkl)
#   · 전체 피처 vs 상위 N개 핵심 피처 재학습 성능 비교 (재학습 없이 읽어 그림)
#   · models_top["HSKR"]가 커스텀 객체 → 언피클 전 hskr_model import 필요
# ─────────────────────────────────────────────────────────────
# 캐시하지 않음 (stale 캐시 방지)
def load_fi():
    """피처 중요도 번들 로드. (bundle|None, err|None) 반환 — 실패해도 앱이 죽지 않도록 방어."""
    base = os.path.dirname(__file__)
    for p in (os.path.join(base, "model", "fi_models.pkl"),
              os.path.join(base, "fi_models.pkl")):
        if os.path.exists(p):
            try:
                from hskr_model import HybridSeasonalKernelRidge  # noqa: F401  (언피클에 필요)
                with open(p, "rb") as f:
                    return pickle.load(f), None
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"
    return None, "파일 없음: model/fi_models.pkl (또는 루트)"


def load_learning_curves():
    """사전 계산된 learning curve 결과 로드 (인앱 학습 없이 그리기 위함).

    캐시하지 않음 — 파일이 가벼워(수십 KB) 매 실행 새로 읽어 파일 변경/추가를 즉시 반영.
    실패 시 (None, 진단메시지) 로 원인을 함께 반환.
    """
    base = os.path.dirname(__file__)
    cands = [os.path.join(base, "model", "learning_curves.pkl"),
             os.path.join(base, "learning_curves.pkl")]
    for p in cands:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), None
            except Exception as e:
                return None, f"로드 오류({os.path.basename(p)}): {type(e).__name__} — {e}"
    return None, "파일 없음: " + " / ".join(cands)


# 캐시하지 않음 — pkl 구조 변경 시 stale 캐시로 KeyError 방지 (파일 3.7K, 매번 읽기 무방)
def load_model_compare():
    """모델 예측 페이지용 model_compare.pkl 로드. (obj|None, err|None)."""
    base = os.path.dirname(__file__)
    for p in (os.path.join(base, "model", "model_compare.pkl"),
              os.path.join(base, "model_compare.pkl")):
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), None
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"
    return None, "파일 없음: model/model_compare.pkl"


def _plot_lc(name, d, col):
    """Learning curve 1개를 col에 그림 (모델 예측·진단 공용 모듈레벨 헬퍼)."""
    ts = np.asarray(d["train_sizes"], float)
    tr = np.asarray(d["train_scores"], float); va = np.asarray(d["val_scores"], float)
    tr_m = tr.mean(1) if tr.ndim == 2 else tr
    va_m = va.mean(1) if va.ndim == 2 else va
    va_s = va.std(1) if va.ndim == 2 else np.zeros_like(va_m)
    flc = go.Figure()
    flc.add_trace(go.Scatter(x=np.concatenate([ts, ts[::-1]]),
                             y=np.concatenate([va_m + va_s, (va_m - va_s)[::-1]]),
                             fill="toself", fillcolor="rgba(0,102,204,0.12)",
                             line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    flc.add_trace(go.Scatter(x=ts, y=tr_m, name="학습", mode="lines+markers",
                             line=dict(color=TOK["ink"], width=2)))
    flc.add_trace(go.Scatter(x=ts, y=va_m, name="검증", mode="lines+markers",
                             line=dict(color=TOK["primary"], width=2.4)))
    flc.update_layout(title=name, height=320, xaxis_title="학습 표본수", yaxis_title="R²",
                      legend=dict(orientation="h", y=1.16))
    style_fig(flc)
    col.plotly_chart(flc, use_container_width=True, config={"displayModeBar": False})


def lime_explain(predict_fn, x_row, X_bg, feature_names, n_samples=600, n_top=12, seed=0):
    """경량 LIME(추가 의존성 0): x_row 주변 perturbation→거리가중 Ridge 국소 선형근사.
    반환: DataFrame(변수, 국소기여) 내림차순(|기여|), 상위 n_top개."""
    from sklearn.linear_model import Ridge as _Ridge
    rng = np.random.default_rng(seed)
    x = np.asarray(x_row, float)
    sd = np.asarray(X_bg, float).std(0); sd[sd == 0] = 1.0
    Z = rng.normal(x, sd, size=(n_samples, len(x)))
    Z[0] = x
    yz = np.asarray(predict_fn(Z), float)
    dist = np.sqrt((((Z - x) / sd) ** 2).sum(1))
    w = np.exp(-(dist ** 2) / (2 * (np.median(dist) + 1e-9) ** 2))
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    loc = _Ridge(alpha=1.0).fit(Zs, yz, sample_weight=w)
    out = pd.DataFrame({"변수": list(feature_names), "국소기여": loc.coef_})
    return out.reindex(out["국소기여"].abs().sort_values(ascending=False).index).head(n_top).reset_index(drop=True)


@st.cache_data
def compute_cca_vif(_df, cols, target="총이용객", vif_threshold=10.0, key=None):
    """VIF로 공선성 제거 후, 각 피처 vs target 의 CCA(1성분) + 상관(피어슨/스피어만/켄달).

    원본 run_cca 로직을 대시보드용으로 이식 (VIF 적용 단일, 전후비교 없음).
    반환: (corr_df, variates{피처:(xc,yc,r,p)}, vif_cols)
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    keep = [c for c in cols if c in _df.columns and _df[c].std() > 0]
    while len(keep) > 1:                          # Stepwise VIF
        Xc = sm.add_constant(_df[keep].astype(float).fillna(0))
        vifs = [variance_inflation_factor(Xc.values, i + 1) for i in range(len(keep))]
        if max(vifs) <= vif_threshold:
            break
        keep.pop(int(np.argmax(vifs)))
    vif_cols = keep

    ys = StandardScaler().fit_transform(_df[[target]].astype(float))
    rows, variates = [], {}
    for j in vif_cols:
        xs = StandardScaler().fit_transform(_df[[j]].astype(float).fillna(0))
        cca = CCA(scale=False, n_components=1).fit(xs, ys)
        xc, yc = cca.transform(xs, ys)
        xc, yc = xc[:, 0], yc[:, 0]
        rP, pP = stats.pearsonr(xc, yc)
        rS, pS = stats.spearmanr(xc, yc)
        rK, pK = stats.kendalltau(xc, yc)
        rows.append({"X_피처": j, "Pearson_r": round(float(rP), 3), "Pearson_p": round(float(pP), 3),
                     "Spearman_r": round(float(rS), 3), "Spearman_p": round(float(pS), 3),
                     "Kendall_r": round(float(rK), 3), "Kendall_p": round(float(pK), 3),
                     "유의": "✅" if pP < 0.05 else ""})
        variates[j] = (xc.tolist(), yc.tolist(), float(rP), float(pP))
    corr_df = pd.DataFrame(rows).sort_values("Pearson_r", ascending=False).reset_index(drop=True)
    return corr_df, variates, vif_cols


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
        font=dict(family='Pretendard Variable, Pretendard, -apple-system, sans-serif', color=ink, size=13),
        margin=dict(l=20, r=20, t=40, b=30),
        title=dict(font=dict(family='Pretendard Variable, Pretendard, -apple-system, sans-serif', size=20, color=ink), x=0.02),
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
        <span style="font-weight:600; letter-spacing:-0.01em">데이터 분석 응용</span>
      </div>
      <div class="nav-links">
        <span title="한강공원" style="display:flex">{icon('wave', 16, TOK['on_dark'], TOK['primary_on_dark'])}</span>
        <span title="분석 대시보드" style="display:flex">{icon('chart', 16, TOK['on_dark'], TOK['primary_on_dark'])}</span>
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
    # 과거 <section> 배경 타일은 Streamlit에서 내용을 못 감싸고 '빈 색상 박스'만 남겨
    # 디자인을 해쳤음. → 배경 띠 없이 앵커 + 상단 여백만 두고 모든 콘텐츠는 단일 배경 위에 렌더.
    a = f' id="{anchor}"' if anchor else ""
    st.markdown(f'<div{a} style="height:4px"></div>', unsafe_allow_html=True)


def tile_close() -> None:
    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)


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
    # (미니 지도는 개요 상단 헤더로 이동됨)

    PAGES = [
        ("개요",            "spark"),
        ("EDA",            "chart"),
        ("t-test & VIF",   "scatter"),
        ("모델 예측",       "model"),
        ("신규 모델 (HSKR)", "spark"),
        ("핵심 변수 선별 효과", "model"),
        ("예측 시뮬레이터",  "boot"),
        ("Conformal",      "interval"),
        ("Bootstrap CI",   "boot"),
        ("Nested CV",      "cv"),
    ]
    page = st.radio("분석", [p[0] for p in PAGES], key="page", label_visibility="visible")

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="caption" style="line-height:1.7">
      <div class="body-strong" style="color:var(--ink); margin-bottom:6px">데이터 요약</div>
      공원-월 표본 · {len(pm):,}<br/>
      관측 월수 · {len(monthly):,}<br/>
      피처 수 · {len(num_cols)}<br/>
      모델 · RandomForest
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Global nav
# ─────────────────────────────────────────────────────────────
render_global_nav()


# ─────────────────────────────────────────────────────────────
# Hero tile — light
# ─────────────────────────────────────────────────────────────
tile_open("light", anchor="overview")
st.markdown("""
<div style="text-align:center; max-width:920px; margin:8px auto 0 auto">
  <h1 class="h-hero">한강공원 이용객을, 데이터로.</h1>
  <p class="lead">EDA부터 SHAP, Conformal, Bootstrap, Nested CV까지 — 하나의 워크플로우.</p>
</div>
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

# ── 헤더 행: 미니맵(2칸) + 공원/분석 선택 카드 (클릭 가능)
PAGE_DESC = {
    "개요": "전체 요약 · 한눈에 보기", "EDA": "탐색적 데이터 분석",
    "t-test & VIF": "피처 선별 · 공선성", "모델 예측": "비교1 · VIF 모델 비교(업로드 결과)",
    "예측 시뮬레이터": "입력 → 실시간 예측", "Conformal": "예측 구간",
    "Bootstrap CI": "성능 신뢰구간", "Nested CV": "일반화 추정",
    "신규 모델 (HSKR)": "HSKR vs Ridge · 11공원",
    "핵심 변수 선별 효과": "비교2 · VIF+중요도 + SHAP/LIME/잔차",
}


def _hdr_pick_park():
    st.session_state["_map_pick"] = st.session_state["hdr_park"]


def _hdr_pick_page():
    st.session_state["_nav"] = st.session_state["hdr_page"]


st.markdown("""
<style>
[class*="st-key-hdrcard"]{box-sizing:border-box;background:#fff;border:1px solid var(--hairline);
  border-radius:18px;padding:14px 18px;height:150px;display:flex;flex-direction:column;
  justify-content:space-between;transition:border-color .14s,box-shadow .14s;}
[class*="st-key-hdrcard"]:hover{border-color:var(--primary);box-shadow:var(--product-shadow);}
[class*="st-key-hdrcard"] [data-baseweb="select"]>div{border:none!important;background:transparent!important;
  padding-left:0!important;box-shadow:none!important;}
[class*="st-key-hdrcard"] [data-baseweb="select"]>div>div{font-size:22px!important;font-weight:600!important;
  color:var(--ink)!important;letter-spacing:-0.02em;}
[class*="st-key-hdrmap"]{box-sizing:border-box;height:150px;border-radius:18px;overflow:hidden;
  border:1px solid var(--hairline);}
[class*="st-key-hdrmap"] iframe{height:148px!important;min-height:148px!important;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

# 헤더 위젯 상태를 현재 선택과 동기화 (위젯 생성 전)
if park_list and st.session_state.get("hdr_park") != selected_park:
    st.session_state["hdr_park"] = selected_park if selected_park in park_list else park_list[0]
if st.session_state.get("hdr_page") != page:
    st.session_state["hdr_page"] = page

hcol = st.columns([2, 1, 1], gap="medium")
with hcol[0]:
    with st.container(key="hdrmap"):
        _hm = folium.Map(location=[37.53, 126.98], zoom_start=11, tiles="CartoDB positron",
                         zoom_control=False, scrollWheelZoom=False, dragging=False)
        for _n, _c in PARK_COORDS.items():
            _s = (_n == selected_park)
            folium.CircleMarker(location=_c, radius=8 if _s else 5,
                                color="#E8505B" if _s else "#0066cc", weight=2, fill=True,
                                fill_color="#E8505B" if _s else "#0066cc", fill_opacity=0.95,
                                tooltip=_n, popup=_n).add_to(_hm)
        _hd = st_folium(_hm, height=150, use_container_width=True, key="header_map")
        if _hd:
            _o = _hd.get("last_object_clicked") or _hd.get("last_clicked")
            if _o:
                _np = nearest_park(_o["lat"], _o["lng"])
                if _np != selected_park and _np in (park_list or [_np]):
                    st.session_state["_map_pick"] = _np
                    st.rerun()
with hcol[1]:
    with st.container(key="hdrcard_park"):
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
                    f'{icon("wave", 18, TOK["ink_48"], TOK["primary"])}'
                    f'<span class="caption" style="margin:0">선택 공원</span></div>', unsafe_allow_html=True)
        if park_list:
            st.selectbox("공원", park_list, key="hdr_park", on_change=_hdr_pick_park,
                         label_visibility="collapsed")
        _pa = pm[pm["공원명"] == selected_park]["총이용객"].mean() if selected_park in (park_list or []) else float("nan")
        st.markdown(f'<div class="caption" style="margin-top:2px">월평균 {_pa/1e4:.1f}만 명</div>'
                    if _pa == _pa else '<div class="caption">—</div>', unsafe_allow_html=True)
with hcol[2]:
    with st.container(key="hdrcard_page"):
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
                    f'{icon("chart", 18, TOK["ink_48"], TOK["primary"])}'
                    f'<span class="caption" style="margin:0">현재 분석</span></div>', unsafe_allow_html=True)
        st.selectbox("분석", [p[0] for p in PAGES], key="hdr_page", on_change=_hdr_pick_page,
                     label_visibility="collapsed")
        st.markdown(f'<div class="caption" style="margin-top:2px">{PAGE_DESC.get(page, "")}</div>',
                    unsafe_allow_html=True)

selected_park = st.session_state.get("selected_park")
tile_close()


# ─────────────────────────────────────────────────────────────
# Page routing — alternating tiles
# ─────────────────────────────────────────────────────────────

if page == "개요":
    # Dark tile — features grid
    tile_open("light", anchor="features")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--ink)">하나의 대시보드, 아홉 가지 분석.</h2>
    <p class="lead">카드를 누르면 해당 분석으로 바로 이동합니다.</p>
    """, unsafe_allow_html=True)

    # 클릭 가능한 카드 스타일 (keyed container) — 라이트 카드
    st.markdown("""
    <style>
    [class*="st-key-navcard"] {
      background: #ffffff;
      border: 1px solid var(--hairline);
      border-radius: 18px;
      padding: 22px 22px 6px 22px;
      height: 100%;
      transition: transform .14s ease, border-color .14s ease, box-shadow .14s ease;
    }
    [class*="st-key-navcard"]:hover {
      transform: translateY(-4px);
      border-color: var(--primary);
      box-shadow: var(--product-shadow);
    }
    [class*="st-key-navcard"] .stButton > button {
      background: transparent !important;
      color: var(--primary) !important;
      border: none;
      border-top: 1px solid var(--hairline);
      border-radius: 0;
      padding: 12px 0 6px 0;
      margin-top: 12px;
      font-size: 14px;
      font-weight: 600;
      justify-content: flex-start;
    }
    [class*="st-key-navcard"] .stButton > button:hover { color: var(--primary-focus) !important; }
    [class*="st-key-navcard"] .stButton > button:focus { outline: none; box-shadow: none; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    FEATURES = [
        ("EDA",            "chart",    "월별 추이 · 계절성 · 변수 분포"),
        ("t-test & VIF",   "scatter",  "유의 피처 선별 · 공선성 제거"),
        ("모델 예측",       "model",    "비교1: VIF 적용 모델 비교 + 학습곡선"),
        ("신규 모델 (HSKR)", "spark",    "내가 만든 HSKR vs 기존 · 11공원"),
        ("핵심 변수 선별 효과", "model",  "비교2: VIF+중요도(1등 모델) + 진단"),
        ("예측 시뮬레이터",  "boot",     "입력 조정 → 실시간 예측"),
        ("Conformal",      "interval", "분포가정 없는 예측구간"),
        ("Bootstrap CI",   "boot",     "성능지표 신뢰구간"),
        ("Nested CV",      "cv",       "과적합 없는 일반화 추정"),
    ]
    for r in range(0, len(FEATURES), 3):
        cols = st.columns(3, gap="medium")
        for j, (title, ic, desc) in enumerate(FEATURES[r:r + 3]):
            i = r + j
            with cols[j]:
                with st.container(key=f"navcard_{i}"):
                    st.markdown(
                        f'<div style="margin-bottom:12px">{icon(ic, 26, TOK["ink"], TOK["primary"])}</div>'
                        f'<div class="body-strong" style="color:var(--ink)">{title}</div>'
                        f'<div class="caption" style="color:var(--ink-48); margin-top:6px; '
                        f'line-height:1.5; min-height:38px">{desc}</div>',
                        unsafe_allow_html=True)
                    if st.button("열기 →", key=f"nav_{i}", use_container_width=True):
                        st.session_state["_nav"] = title
                        st.rerun()
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
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

    # Parchment tile — 월별 추이 (면적 + 이동평균)
    tile_open("parchment", anchor="sample")
    st.markdown('<h2 class="h-section">월별 이용객 추이 (전체 합계)</h2>', unsafe_allow_html=True)
    _m = monthly.sort_values("연월")
    _roll = _m["총이용객"].rolling(12, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_m["연월"], y=_m["총이용객"], name="월 합계", mode="lines",
                             line=dict(color=TOK["primary"], width=2.4),
                             fill="tozeroy", fillcolor="rgba(0,102,204,0.08)"))
    fig.add_trace(go.Scatter(x=_m["연월"], y=_roll, name="12개월 이동평균", mode="lines",
                             line=dict(color=TOK["ink"], width=1.6, dash="dot")))
    fig.update_layout(height=400, hovermode="x unified",
                      xaxis=dict(rangeslider=dict(visible=True), title=""))
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

    SEASON_C = {"봄": "#34c759", "여름": "#0a84ff", "가을": "#ff9f0a", "겨울": "#5e5ce6"}
    SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
    sel_rows = pm[pm["공원명"] == selected_park]
    has_sel = len(sel_rows) > 0
    st.markdown(f'<div class="caption" style="margin:-8px 0 14px 0">비교 차트는 사이드바에서 선택한 '
                f'<b style="color:var(--primary)">{selected_park}</b> 기준입니다.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["이용 추이", "공원 비교", "계절 패턴", "히트맵", "상관·구성",
                    "기술통계", "공원별 EDA", "CCA"])

    # ── 탭1: 이용 추이 (전체 + 월별 전체 vs 선택) ───────────────
    with tabs[0]:
        m = monthly.sort_values("연월")
        roll = m["총이용객"].rolling(12, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["연월"], y=m["총이용객"], name="월 합계", mode="lines",
                                 line=dict(color=TOK["primary"], width=2.4),
                                 fill="tozeroy", fillcolor="rgba(0,102,204,0.08)"))
        fig.add_trace(go.Scatter(x=m["연월"], y=roll, name="12개월 이동평균", mode="lines",
                                 line=dict(color=TOK["ink"], width=1.6, dash="dot")))
        pk = m.loc[m["총이용객"].idxmax()]
        fig.add_annotation(x=pk["연월"], y=pk["총이용객"], text=f"최대 {pk['총이용객']/1e6:.1f}M",
                           showarrow=True, arrowhead=2, ax=0, ay=-32, font=dict(color=TOK["primary"]))
        fig.update_layout(title="월별 총이용객 추이 (전체 합계)", height=420, hovermode="x unified",
                          xaxis=dict(rangeslider=dict(visible=True), title=""))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # 월별 평균 — 전체 vs 선택 공원
        all_m = pm.groupby("월")["총이용객"].mean().reindex(range(1, 13))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[f"{i}월" for i in range(1, 13)], y=all_m.values,
                                  name="전체 공원 평균", mode="lines+markers",
                                  line=dict(color="#94A3B8", width=2, dash="dot")))
        if has_sel:
            sel_m = sel_rows.groupby("월")["총이용객"].mean().reindex(range(1, 13))
            fig2.add_trace(go.Scatter(x=[f"{i}월" for i in range(1, 13)], y=sel_m.values,
                                      name=selected_park, mode="lines+markers",
                                      line=dict(color=TOK["primary"], width=3)))
        fig2.update_layout(title="월별 평균 이용객 — 전체 vs 선택 공원", height=380,
                           hovermode="x unified", yaxis_title="평균 이용객")
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── 탭2: 공원 비교 (랭킹 + 추이 오버레이) ───────────────────
    with tabs[1]:
        rank = pm.groupby("공원명")["총이용객"].mean().sort_values().reset_index()
        colors = [("#E8505B" if p == selected_park else TOK["primary"]) for p in rank["공원명"]]
        figr = go.Figure(go.Bar(x=rank["총이용객"], y=rank["공원명"], orientation="h",
                                marker_color=colors,
                                text=[f"{v/1e4:.0f}만" for v in rank["총이용객"]], textposition="outside"))
        figr.update_layout(title=f"공원별 월평균 이용객 랭킹 (빨강 = {selected_park})",
                           height=460, xaxis_title="월평균 이용객")
        style_fig(figr)
        st.plotly_chart(figr, use_container_width=True, config={"displayModeBar": False})

        fig = go.Figure()
        for p in park_list:
            d = pm[pm["공원명"] == p].sort_values("연월")
            sel = (p == selected_park)
            fig.add_trace(go.Scatter(
                x=d["연월"], y=d["총이용객"], name=p, mode="lines",
                line=dict(color=TOK["primary"] if sel else "rgba(140,140,150,0.45)",
                          width=3.2 if sel else 1.2),
                hovertemplate=f"{p}<br>%{{x|%Y-%m}} · %{{y:,.0f}}<extra></extra>", showlegend=sel))
        fig.update_layout(title=f"공원별 월 이용객 추이 (강조: {selected_park})", height=460, hovermode="closest")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 탭3: 계절 패턴 (전체 vs 선택 바 + 선택 공원 비율 + 분포) ──
    with tabs[2]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            all_s = pm.groupby("계절")["총이용객"].mean().reindex(SEASON_ORDER)
            figs = go.Figure()
            figs.add_trace(go.Bar(name="전체 평균", x=SEASON_ORDER, y=all_s.values,
                                  marker_color="#94A3B8",
                                  text=[f"{v/1e4:.0f}만" for v in all_s.values], textposition="outside"))
            if has_sel:
                sel_s = sel_rows.groupby("계절")["총이용객"].mean().reindex(SEASON_ORDER)
                figs.add_trace(go.Bar(name=selected_park, x=SEASON_ORDER, y=sel_s.values,
                                      marker_color=TOK["primary"],
                                      text=[f"{v/1e4:.0f}만" for v in sel_s.values], textposition="outside"))
            figs.update_layout(title="계절별 이용객 — 전체 vs 선택", barmode="group", height=400)
            style_fig(figs)
            st.plotly_chart(figs, use_container_width=True, config={"displayModeBar": False})
        with c2:
            if has_sel:
                pie_df = sel_rows.groupby("계절")["총이용객"].sum().reindex(SEASON_ORDER).reset_index()
                figp = px.pie(pie_df, names="계절", values="총이용객", color="계절",
                              color_discrete_map=SEASON_C, hole=0.45)
                figp.update_traces(textinfo="percent+label", textfont_size=13)
                figp.update_layout(title=f"{selected_park} 계절별 비율", height=400, showlegend=False)
                style_fig(figp)
                st.plotly_chart(figp, use_container_width=True, config={"displayModeBar": False})

        figb = px.box(pm, x="계절", y="총이용객", color="계절",
                      category_orders={"계절": SEASON_ORDER}, color_discrete_map=SEASON_C,
                      points="all")
        figb.update_layout(title="계절별 이용객 분포 (공원-월)", height=420,
                           xaxis=dict(title="계절"), yaxis_title="이용객", showlegend=False)
        style_fig(figb)
        st.plotly_chart(figb, use_container_width=True, config={"displayModeBar": False})

    # ── 탭4: 히트맵 (공원×계절, 공원×월) ────────────────────────
    with tabs[3]:
        piv_s = pm.pivot_table(index="공원명", columns="계절", values="총이용객", aggfunc="mean").reindex(columns=SEASON_ORDER)
        piv_s = piv_s.loc[piv_s.mean(axis=1).sort_values(ascending=False).index]
        figh = px.imshow(piv_s / 1e4, color_continuous_scale="Blues", aspect="auto", text_auto=".0f",
                         labels=dict(x="계절", y="공원", color="평균(만명)"))
        if selected_park in piv_s.index:
            ri = list(piv_s.index).index(selected_park)
            figh.add_shape(type="rect", x0=-0.5, x1=len(SEASON_ORDER) - 0.5, y0=ri - 0.5, y1=ri + 0.5,
                           line=dict(color="#E8505B", width=3))
        figh.update_layout(title=f"공원 × 계절 평균 이용객 (만 명, 빨강 = {selected_park})", height=460)
        style_fig(figh)
        st.plotly_chart(figh, use_container_width=True, config={"displayModeBar": False})

        piv_m = pm.pivot_table(index="공원명", columns="월", values="총이용객", aggfunc="mean").reindex(columns=range(1, 13))
        piv_m = piv_m.loc[piv_m.mean(axis=1).sort_values(ascending=False).index]
        figm = px.imshow(piv_m / 1e4, color_continuous_scale="Blues", aspect="auto", text_auto=".0f",
                         labels=dict(x="월", y="공원", color="평균(만명)"), x=[f"{i}월" for i in range(1, 13)])
        figm.update_layout(title="공원 × 월 평균 이용객 (만 명)", height=460)
        style_fig(figm)
        st.plotly_chart(figm, use_container_width=True, config={"displayModeBar": False})

    # ── 탭5: 상관 · 시간대 구성 ─────────────────────────────────
    with tabs[4]:
        cc = [c for c in feature_cols if c not in ("월sin", "월cos") and c in pm.columns]
        top = pm[cc].corrwith(pm["총이용객"]).abs().sort_values(ascending=False).head(10).index.tolist()
        corr = pm[top + ["총이용객"]].corr()
        figc = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
        figc.update_layout(title="피처 상관관계 (타깃 상관 상위 10 + 총이용객)", height=520)
        style_fig(figc)
        st.plotly_chart(figc, use_container_width=True, config={"displayModeBar": False})

        dl = monthly.melt(id_vars="연월", value_vars=time_cols, var_name="시간대", value_name="이용객")
        figt = px.area(dl, x="연월", y="이용객", color="시간대", groupnorm="fraction",
                       color_discrete_sequence=[TOK["primary"], TOK["primary_on_dark"], "#86868b"])
        figt.update_layout(title="시간대별 이용 비중 (아침·낮·저녁)", height=400, yaxis=dict(tickformat=".0%"))
        style_fig(figt)
        st.plotly_chart(figt, use_container_width=True, config={"displayModeBar": False})

    # ── 탭6: 기술통계 (전체 + 공원별) ───────────────────────────
    with tabs[5]:
        st.markdown('<div class="caption" style="margin-bottom:8px">월 단위 기술통계 '
                    '(count·mean·std·min·25/50/75%·max)</div>', unsafe_allow_html=True)
        st.markdown('**전체 (공원-월 표본)**')
        desc_cols = ["총이용객"] + [c for c in feature_cols if c not in ("월sin", "월cos")][:8]
        desc_all = pm[desc_cols].describe().T.round(1).reset_index().rename(columns={"index": "변수"})
        st.dataframe(desc_all, use_container_width=True, hide_index=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('**공원별 총이용객 요약**')
        g = (pm.groupby("공원명")["총이용객"].describe()[["count", "mean", "std", "min", "50%", "max"]]
             .round(0).reset_index())
        g.columns = ["공원", "관측수", "평균", "표준편차", "최소", "중앙값", "최대"]
        st.dataframe(g.sort_values("평균", ascending=False), use_container_width=True, hide_index=True)

    # ── 탭7: 공원별 EDA (선택 공원 상세) ────────────────────────
    with tabs[6]:
        if has_sel:
            d = sel_rows.sort_values("연월")
            st.markdown(f'**{selected_park}** — 월별 추이 / 계절 / 시간대 구성', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3, gap="medium")
            c1.markdown(metric_card(f"{d['총이용객'].mean()/1e4:.1f}만", "월평균"), unsafe_allow_html=True)
            c2.markdown(metric_card(f"{d['총이용객'].max()/1e4:.1f}만", "최대월"), unsafe_allow_html=True)
            c3.markdown(metric_card(f"{d['검색량'].mean():.0f}", "평균 검색량"), unsafe_allow_html=True)
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=d["연월"], y=d["총이용객"], mode="lines+markers",
                                    line=dict(color=TOK["primary"], width=2.4), fill="tozeroy",
                                    fillcolor="rgba(0,102,204,0.08)", name="총이용객"))
            f1.update_layout(title=f"{selected_park} 월별 이용객", height=360, hovermode="x unified")
            style_fig(f1)
            st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
            cc1, cc2 = st.columns(2, gap="medium")
            with cc1:
                prof = d.groupby("월")["총이용객"].mean().reindex(range(1, 13))
                f2 = go.Figure(go.Scatter(x=[f"{m}월" for m in range(1, 13)], y=prof.values,
                                          mode="lines+markers", line=dict(color=TOK["ink"], width=2)))
                f2.update_layout(title="월별 평균 프로파일", height=320, yaxis_title="평균 이용객")
                style_fig(f2)
                st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
            with cc2:
                f3 = px.box(d, x="계절", y="총이용객", color="계절",
                            category_orders={"계절": SEASON_ORDER}, color_discrete_map=SEASON_C, points="all")
                f3.update_layout(title="계절별 분포", height=320, showlegend=False)
                style_fig(f3)
                st.plotly_chart(f3, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("선택 공원 데이터가 없습니다.")

    # ── 탭8: CCA (피처별 vs 총이용객, VIF 적용) ─────────────────
    with tabs[7]:
        st.markdown('<div class="caption" style="margin-bottom:8px">CCA(정준상관분석): '
                    '<b>VIF 통과 변수군(X)</b> ↔ <b>시간대 변수군(Y: 아침·낮·저녁)</b>. '
                    '1번째 정준쌍 산점도 + X측 변수 기여도, 결과 CSV·PNG 저장.</div>', unsafe_allow_html=True)
        try:
            from sklearn.cross_decomposition import CCA
            from sklearn.preprocessing import StandardScaler
            cca_cols = [c for c in feature_cols if c not in ("월sin", "월cos")]
            corr_df, variates, vif_cols = compute_cca_vif(monthly, cca_cols, key="ALL")
            Ycols = [c for c in time_cols if c in monthly.columns]
            Xcols = [c for c in vif_cols if c not in (Ycols + ["총이용객"])
                     and c in monthly.columns and monthly[c].std() > 0]
            st.markdown(f'<div class="caption">표본 {len(monthly)}개 · '
                        f'X(VIF 변수) <b>{len(Xcols)}개</b> · Y(시간대) <b>{len(Ycols)}개</b></div>',
                        unsafe_allow_html=True)
            if len(Xcols) > len(monthly) // 7:
                st.markdown(f'<div class="caption" style="color:#E8505B">⚠️ X({len(Xcols)})가 표본 대비 '
                            f'많은 편(권장 ≤{len(monthly)//7}) — 1번째 정준상관이 다소 높게 나올 수 있습니다.</div>',
                            unsafe_allow_html=True)
            if len(Xcols) >= 2 and len(Ycols) >= 2:
                Xm = StandardScaler().fit_transform(monthly[Xcols])
                Ym = StandardScaler().fit_transform(monthly[Ycols])
                ncomp = min(len(Xcols), len(Ycols))
                cca = CCA(n_components=ncomp).fit(Xm, Ym)
                U, V = cca.transform(Xm, Ym)
                corrs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(ncomp)]
                comp_cols = [f"CC{i+1}" for i in range(ncomp)]

                kc = st.columns(ncomp, gap="medium")
                for i in range(ncomp):
                    kc[i].markdown(metric_card(f"{corrs[i]:.3f}", f"정준상관 CC{i+1}"), unsafe_allow_html=True)

                m1, b1 = np.polyfit(U[:, 0], V[:, 0], 1)
                xr = np.array([U[:, 0].min(), U[:, 0].max()])
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    f1 = go.Figure()
                    f1.add_trace(go.Scatter(x=U[:, 0], y=V[:, 0], mode="markers", name="표본",
                                            marker=dict(color="#9aa0a6", size=8, line=dict(color="#333", width=1))))
                    f1.add_trace(go.Scatter(x=xr, y=m1 * xr + b1, mode="lines", name="회귀",
                                            line=dict(color="#E8505B", width=2)))
                    f1.update_layout(title=f"1번째 정준쌍 (r={corrs[0]:.2f})", height=360,
                                     xaxis_title="시설 정준변수 U₁", yaxis_title="시간대 정준변수 V₁")
                    style_fig(f1)
                    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
                with c2:
                    xw1 = pd.Series(cca.x_weights_[:, 0], index=Xcols).sort_values(key=abs, ascending=False).head(10)[::-1]
                    f2 = go.Figure(go.Bar(x=xw1.values, y=xw1.index, orientation="h", marker_color=TOK["primary"]))
                    f2.add_vline(x=0, line=dict(color=TOK["ink"], dash="dot"))
                    f2.update_layout(title="X측 변수 기여도 (CC1 상위10)", height=360, xaxis_title="weight")
                    style_fig(f2)
                    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

                yw_df = (pd.DataFrame(np.round(cca.y_weights_, 3), index=Ycols, columns=comp_cols)
                         .reset_index().rename(columns={"index": "시간대"}))
                st.markdown('<div class="caption" style="margin:6px 0">Y측(시간대) 가중치</div>', unsafe_allow_html=True)
                st.dataframe(yw_df, use_container_width=True, hide_index=True)

                # ── 결과 저장 (CSV + 그림 PNG)
                st.markdown('<h3 class="h-section" style="margin-top:18px">결과 저장</h3>', unsafe_allow_html=True)
                corrs_df = pd.DataFrame({"정준쌍": comp_cols, "정준상관계수": [round(c, 4) for c in corrs]})
                xw_full = (pd.DataFrame(np.round(cca.x_weights_, 4), index=Xcols, columns=comp_cols)
                           .reset_index().rename(columns={"index": "X_피처"}))
                d1, d2, d3 = st.columns(3, gap="medium")
                d1.download_button("정준상관 CSV", corrs_df.to_csv(index=False).encode("utf-8-sig"),
                                   "cca_canonical_corr.csv", "text/csv")
                d2.download_button("X 가중치 CSV", xw_full.to_csv(index=False).encode("utf-8-sig"),
                                   "cca_x_weights.csv", "text/csv")
                d3.download_button("Y 가중치 CSV", yw_df.to_csv(index=False).encode("utf-8-sig"),
                                   "cca_y_weights.csv", "text/csv")
                if st.button("그림(PNG) 파일 생성", key="cca_png"):
                    import io as _io, matplotlib.pyplot as _plt
                    figp, axp = _plt.subplots(figsize=(6, 5))
                    axp.scatter(U[:, 0], V[:, 0], c="gray", edgecolor="black")
                    axp.plot(xr, m1 * xr + b1, "r")
                    axp.set_xlabel("시설 정준변수 U1"); axp.set_ylabel("시간대 정준변수 V1")
                    axp.set_title(f"CCA 1번째 정준쌍 (r={corrs[0]:.2f})")
                    b_s = _io.BytesIO(); figp.savefig(b_s, format="png", dpi=150, bbox_inches="tight"); _plt.close(figp)
                    top = pd.Series(cca.x_weights_[:, 0], index=Xcols).sort_values(key=abs, ascending=False).head(10)[::-1]
                    figq, axq = _plt.subplots(figsize=(6, 4))
                    axq.barh(top.index, top.values, color="teal"); axq.axvline(0, color="black", lw=0.5)
                    axq.set_title("X측 변수 기여도 (1번째 정준쌍)")
                    b_w = _io.BytesIO(); figq.savefig(b_w, format="png", dpi=150, bbox_inches="tight"); _plt.close(figq)
                    e1, e2 = st.columns(2, gap="medium")
                    e1.download_button("산점도 PNG", b_s.getvalue(), "cca_scatter.png", "image/png")
                    e2.download_button("기여도 PNG", b_w.getvalue(), "cca_weights.png", "image/png")

                # ── 변수마다 산점도 한 장씩 저장 (예전 run_cca 방식)
                st.markdown('<h3 class="h-section" style="margin-top:18px">변수별 산점도 저장</h3>',
                            unsafe_allow_html=True)
                st.markdown('<div class="caption" style="margin-bottom:6px">각 변수와 총이용객의 CCA 산점도를 '
                            '<b>변수마다 한 장씩(.png)</b> ZIP으로 저장합니다 (+ correlation.csv).</div>',
                            unsafe_allow_html=True)
                only_sig = st.checkbox("유의한 변수만 (p<0.05)", value=True, key="cca_pf_sig")
                if st.button("변수별 산점도 ZIP 생성", key="cca_pf_zip"):
                    import io as _io2, zipfile as _zip, matplotlib.pyplot as _plt2

                    def _cf(s):
                        return s.replace("/", "_").replace(" ", "_").replace(",", "_")

                    items = [(f, v) for f, v in variates.items() if (not only_sig or v[3] < 0.05)]
                    bz = _io2.BytesIO()
                    with st.spinner(f"{len(items)}개 산점도 생성 중..."):
                        with _zip.ZipFile(bz, "w", _zip.ZIP_DEFLATED) as zf:
                            for feat, (xcp, ycp, rp, pp) in items:
                                xa, ya = np.asarray(xcp), np.asarray(ycp)
                                ms, bs = np.polyfit(xa, ya, 1)
                                xrr = np.array([xa.min(), xa.max()])
                                fg, ax = _plt2.subplots(figsize=(6, 6))
                                ax.scatter(xa, ya, color="grey", edgecolor="k")
                                ax.plot(xrr, ms * xrr + bs, color="red")
                                ax.text(xa.min(), ya.max(), f"r={rp:.2f}, p={pp:.3f}",
                                        size="medium", weight="semibold")
                                ax.set_xlabel(feat); ax.set_ylabel("총이용객")
                                ax.set_title(f"CCA Scatter\n{feat} vs 총이용객")
                                pb = _io2.BytesIO()
                                fg.savefig(pb, format="png", dpi=150, bbox_inches="tight")
                                _plt2.close(fg)
                                zf.writestr(f"CCA_{_cf(feat)}.png", pb.getvalue())
                            zf.writestr("correlation.csv", corr_df.to_csv(index=False).encode("utf-8-sig"))
                    st.download_button("⬇️ 산점도 ZIP 다운로드", bz.getvalue(),
                                       "cca_scatter_per_feature.zip", "application/zip")
                    st.success(f"{len(items)}장 생성 완료.")
            else:
                st.warning("X 또는 Y 변수가 부족합니다.")
        except Exception as e:
            st.markdown(f'<div class="caption">CCA 오류 — {e}</div>', unsafe_allow_html=True)
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
            rows.append({"변수": f, "t-stat": float(t), "p-value": float(p),
                         "유의성": "유의" if p < 0.05 else "—"})
        ttest_df = pd.DataFrame(rows).sort_values("p-value")

        sig = ttest_df[ttest_df["유의성"] == "유의"]["변수"].tolist()
        vif = None
        if len(sig) >= 2:
            Xv = monthly[sig].astype(float).fillna(0)
            vif = pd.DataFrame({
                "변수": sig,
                "VIF": [round(variance_inflation_factor(Xv.values, i), 2) for i in range(Xv.shape[1])],
            }).sort_values("VIF", ascending=False)

        # ── 표: t-test 결과 + VIF
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown('<div class="body-strong" style="margin-bottom:10px">t-test 결과</div>', unsafe_allow_html=True)
            disp = ttest_df.assign(**{"t-stat": ttest_df["t-stat"].round(3),
                                      "p-value": ttest_df["p-value"].round(4)})
            st.dataframe(disp, use_container_width=True, hide_index=True)
        with c2:
            if vif is not None:
                st.markdown('<div class="body-strong" style="margin-bottom:10px">VIF (공선성)</div>', unsafe_allow_html=True)
                st.dataframe(vif, use_container_width=True, hide_index=True)
            else:
                st.markdown('<div class="caption">유의 피처가 부족합니다.</div>', unsafe_allow_html=True)

        # ── 시각화 ①: t-test 선택 (−log₁₀ p, 임계선 p=0.05)
        st.markdown('<h2 class="h-section" style="margin-top:26px">① t-test 선택 시각화</h2>', unsafe_allow_html=True)
        st.markdown('<div class="caption" style="margin-bottom:8px">막대가 빨간 점선(p=0.05)을 넘으면 '
                    '<b style="color:var(--primary)">유의 변수로 선택</b>됩니다.</div>', unsafe_allow_html=True)
        tv = ttest_df.copy()
        tv["neglogp"] = -np.log10(np.clip(tv["p-value"].astype(float), 1e-20, 1.0))
        tv = tv.sort_values("neglogp")
        thr = float(-np.log10(0.05))
        colt = [TOK["primary"] if s == "유의" else TOK["ink_48"] for s in tv["유의성"]]
        fig_t = go.Figure(go.Bar(x=tv["neglogp"], y=tv["변수"], orientation="h", marker_color=colt,
                                 text=[f"p={p:.3g}" for p in tv["p-value"]], textposition="outside"))
        fig_t.add_vline(x=thr, line=dict(color="#E8505B", dash="dash"),
                        annotation_text="p=0.05", annotation_position="top")
        fig_t.update_layout(title="변수별 t-test 유의성 (−log₁₀ p · 파랑=선택)", height=max(360, 22 * len(tv)),
                            xaxis_title="−log₁₀(p-value) — 클수록 유의", margin=dict(l=10, r=70))
        style_fig(fig_t)
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

        # ── 시각화 ②: VIF 다중공선성 (임계선 VIF=10)
        if vif is not None:
            st.markdown('<h2 class="h-section" style="margin-top:22px">② VIF 다중공선성 시각화</h2>', unsafe_allow_html=True)
            st.markdown('<div class="caption" style="margin-bottom:8px">유의 변수 중 VIF가 '
                        '<b style="color:#E8505B">10 이상</b>이면 공선성이 커 제거 후보입니다.</div>',
                        unsafe_allow_html=True)
            vf = vif.sort_values("VIF")
            colv = ["#E8505B" if v >= 10 else TOK["primary"] for v in vf["VIF"]]
            fig_v = go.Figure(go.Bar(x=vf["VIF"], y=vf["변수"], orientation="h", marker_color=colv,
                                     text=[f"{v:.1f}" for v in vf["VIF"]], textposition="outside"))
            fig_v.add_vline(x=10, line=dict(color="#E8505B", dash="dash"),
                            annotation_text="VIF=10", annotation_position="top")
            fig_v.update_layout(title="유의 변수 VIF (빨강=10↑ 제거 후보)", height=max(300, 30 * len(vf)),
                                xaxis_title="VIF — 낮을수록 독립적", margin=dict(l=10, r=70))
            style_fig(fig_v)
            st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">분석을 실행할 수 없습니다 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "모델 예측":
    tile_open("light", anchor="model")
    st.markdown("""
    <h2 class="h-display" style="color:var(--ink)">비교 1 — VIF 적용 · 모델 비교</h2>
    <p class="lead">다중공선성(VIF)을 제거한 피처로 학습한 모델들의 성능을 비교해 최고 표준 ML 모델을 고릅니다.
    수치는 업로드한 분석 결과(fi_models.pkl) 그대로입니다.</p>
    """, unsafe_allow_html=True)
    tile_close()

    FB, fi_err = load_fi()
    MC, _ = load_model_compare()
    tile_open("light")
    st.markdown('<h2 class="h-section">VIF 적용 피처로 학습한 모델 성능</h2>', unsafe_allow_html=True)
    if FB is None:
        st.markdown(f"""
        <div class="card">
          <div class="body-strong">분석 결과(fi_models.pkl)를 불러올 수 없습니다</div>
          <div class="caption" style="margin-top:8px; line-height:1.7">
            <b>model/fi_models.pkl</b>이 필요합니다.<br/>진단: <code>{fi_err}</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        mf = FB["metrics_full"]
        allf = list(FB["all_features"])
        order = [m for m in ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest", "ExtraTrees", "HSKR")
                 if m in mf] + [m for m in mf if m not in
                                ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest", "ExtraTrees", "HSKR")]
        std = [m for m in order if m != "HSKR"]
        best = max(std, key=lambda m: mf[m]["R2"]) if std else order[0]
        has_h = "HSKR" in mf

        def _col(m):
            return "#E8505B" if m == "HSKR" else (TOK["primary"] if m == best else TOK["ink_48"])

        k1, k2, k3 = st.columns(3, gap="medium")
        k1.markdown(metric_card(f"{len(allf)}개", "VIF 적용 피처 수"), unsafe_allow_html=True)
        k2.markdown(metric_card(best, f"최고 표준 ML · R² {mf[best]['R2']:.3f}"), unsafe_allow_html=True)
        if has_h:
            k3.markdown(metric_card(f"{mf['HSKR']['R2']:.3f}", "HSKR(신규모델) R²", delta="+신규 모델 페이지"),
                        unsafe_allow_html=True)
        else:
            k3.markdown(metric_card(f"{mf[best]['RMSE']/1e4:.1f}만", f"{best} RMSE"), unsafe_allow_html=True)
        st.markdown('<div class="caption" style="margin:10px 0 16px 0;line-height:1.6">수치 = 업로드한 '
                    '<b>fi_models.pkl</b>(VIF 적용 24개 피처) 그대로. 표준 ML 중 <b style="color:var(--primary)">'
                    f'{best}</b>가 1등. <b style="color:#E8505B">HSKR</b>은 직접 만든 신규 모델로 '
                    '신규 모델 페이지에서 별도 비교합니다.</div>', unsafe_allow_html=True)

        ca, cb = st.columns(2, gap="medium")
        with ca:
            fr = go.Figure(go.Bar(x=[mf[m]["R2"] for m in order], y=order, orientation="h",
                                  marker_color=[_col(m) for m in order],
                                  text=[f"{mf[m]['R2']:.3f}" for m in order], textposition="outside"))
            fr.update_layout(title="R² (높을수록 우수 · 파랑=최고 표준ML)", height=360, xaxis_title="R²", margin=dict(r=70))
            style_fig(fr)
            st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})
        with cb:
            fe = go.Figure(go.Bar(x=[mf[m]["RMSE"] / 1e4 for m in order], y=order, orientation="h",
                                  marker_color=[_col(m) for m in order],
                                  text=[f"{mf[m]['RMSE']/1e4:.1f}" for m in order], textposition="outside"))
            fe.update_layout(title="RMSE(만, 낮을수록 우수)", height=360, xaxis_title="RMSE(만)", margin=dict(r=70))
            style_fig(fe)
            st.plotly_chart(fe, use_container_width=True, config={"displayModeBar": False})

        cmp_df = pd.DataFrame([{"모델": m, "R²": round(mf[m]["R2"], 3),
                                "RMSE(만)": round(mf[m]["RMSE"] / 1e4, 1),
                                "비고": "신규모델" if m == "HSKR" else ("★ 최고 표준ML" if m == best else "")}
                               for m in order])
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ 모델 성능표 CSV", cmp_df.to_csv(index=False).encode("utf-8-sig"),
                           "model_compare.csv", "text/csv")
        st.markdown(f'<div class="caption" style="margin-top:8px">→ 최고 표준 ML '
                    f'<b style="color:var(--primary)">{best}</b>은 <b>핵심 변수 선별 효과</b>(비교 2)에서 '
                    f'VIF+중요도 축소·진단에 사용됩니다.</div>', unsafe_allow_html=True)

        if MC and MC.get("lc"):
            st.markdown('<h2 class="h-section" style="margin-top:30px">모델별 Learning Curve</h2>',
                        unsafe_allow_html=True)
            st.markdown('<div class="caption" style="margin-bottom:10px">훈련 표본 수를 늘려가며 학습·검증 R²를 봅니다. '
                        '두 곡선이 수렴하면 데이터 충분, 갭이 크면 과적합.</div>', unsafe_allow_html=True)
            lcs = MC["lc"]
            names = list(lcs.keys())
            for r in range(0, len(names), 3):
                rowm = names[r:r + 3]
                lcols = st.columns(len(rowm), gap="medium")
                for nm, col in zip(rowm, lcols):
                    _plot_lc(nm, lcs[nm], col)
            st.caption("※ 학습곡선은 동일 24개 피처로 앱에서 사전계산(참고용·곡선 형태 기준). "
                       "성능 수치(위 표)는 업로드한 fi_models.pkl 기준이라 절대 R²는 다를 수 있습니다.")
    tile_close()


elif page == "신규 모델 (HSKR)":
    tile_open("light", anchor="hskr")
    st.markdown('<h2 class="h-display" style="color:var(--ink)">신규 HSKR vs 기존 Ridge</h2>',
                unsafe_allow_html=True)
    st.markdown('<p class="lead">직접 구현한 Hybrid Seasonal Kernel Ridge(계절 푸리에 + RBF 커널)와 '
                '기존 Ridge 모델을 공원별로 비교합니다 — 전 11개 공원.</p>', unsafe_allow_html=True)
    tile_close()

    B, hskr_err = load_hskr()
    tile_open("light")
    if B is None:
        diag = (f'<br/><br/>진단: <code>{hskr_err}</code>' if hskr_err else "")
        st.markdown(f"""
        <div class="card">
          <div class="body-strong">HSKR 번들을 불러올 수 없습니다</div>
          <div class="caption" style="margin-top:8px; line-height:1.7">
            아래 두 파일을 두고 다시 실행하세요:<br/>
            · <b>hskr_model.py</b> → 레포 루트 (appnew.py와 같은 폴더)<br/>
            · <b>hskr_model.pkl</b> → <b>model/hskr_model.pkl</b> (또는 루트)<br/>
            <i>ModuleNotFoundError(numpy/sklearn)면 requirements.txt 버전 고정이 필요합니다.</i>{diag}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        parks = list(B["parks"])
        core = [p for p in ("망원한강공원", "이촌한강공원", "잠실한강공원") if p in parks]
        bn = "Ridge" if "Ridge" in B["per_park"][parks[0]]["metrics"] else B["best_base_name"]

        def _red(p):
            mm = B["per_park"][p]["metrics"]
            return (mm[bn]["RMSE"] - mm["HSKR"]["RMSE"]) / mm[bn]["RMSE"] * 100 if mm[bn]["RMSE"] else 0.0

        # ── 헤드라인: 전 11개 공원 전체(풀링) 성능
        yt_all = np.concatenate([np.asarray(B["per_park"][p]["y_test"], float) for p in parks])
        hk_all = np.concatenate([np.asarray(B["per_park"][p]["hskr_pred_test"], float) for p in parks])
        rg_all = np.concatenate([np.asarray(B["per_park"][p]["base_pred_test"][bn], float) for p in parks])
        hk_r2, hk_rmse = r2_score(yt_all, hk_all), mean_squared_error(yt_all, hk_all) ** 0.5
        rg_r2, rg_rmse = r2_score(yt_all, rg_all), mean_squared_error(yt_all, rg_all) ** 0.5
        ov_red = (rg_rmse - hk_rmse) / rg_rmse * 100 if rg_rmse else 0.0
        n_win = sum(1 for p in parks if _red(p) > 0)

        st.markdown(f'<div class="caption" style="margin-bottom:8px;line-height:1.6">HSKR(계절+커널)을 '
                    f'<b>전 {len(parks)}개 공원</b>에 공원별로 학습해 기존 Ridge와 비교합니다. '
                    f'평가지표는 <b>R²</b>와 <b>RMSE</b>. HSKR이 R²·RMSE 모두에서 Ridge를 앞서며 '
                    f'<b>{n_win}/{len(parks)}개 공원</b>에서 RMSE를 낮췄습니다. 아래 전체값은 테스트 구간 풀링이라 '
                    f'공원 간 스케일 차(수만~수백만)로 R²가 낮게 나옵니다(공원별 R²는 아래 상세 참고).</div>',
                    unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3, gap="medium")
        g1.markdown(metric_card(f"{hk_r2:.3f}", "전체 R² (HSKR)", delta=f"+Ridge {rg_r2:.3f}"), unsafe_allow_html=True)
        g2.markdown(metric_card(f"{hk_rmse/1e4:.1f}만", "전체 RMSE (HSKR)",
                                delta=f"+Ridge {rg_rmse/1e4:.1f}만"), unsafe_allow_html=True)
        g3.markdown(metric_card(f"+{ov_red:.1f}%", "RMSE 감소율", delta="+vs Ridge"), unsafe_allow_html=True)
        fov = go.Figure(go.Bar(x=[hk_rmse / 1e4, rg_rmse / 1e4], y=["HSKR", "Ridge(기존)"], orientation="h",
                               marker_color=[TOK["primary"], "#E8505B"],
                               text=[f"{hk_rmse/1e4:.1f}만", f"{rg_rmse/1e4:.1f}만"], textposition="outside"))
        fov.update_layout(title=f"{len(parks)}개 공원 전체 RMSE (낮을수록 우수)", height=240, xaxis_title="RMSE(만)",
                          yaxis=dict(autorange="reversed"))
        style_fig(fov)
        st.plotly_chart(fov, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<hr style="border:none;border-top:1px solid var(--hairline);margin:22px 0 6px 0">',
                    unsafe_allow_html=True)
        st.markdown('<h2 class="h-section">공원별 상세</h2>', unsafe_allow_html=True)

        # 공원 선택 (상세)
        d_idx = parks.index(selected_park) if selected_park in parks else (parks.index(core[0]) if core else 0)
        cpark = st.selectbox("공원 (전 11개 공원)", parks, index=d_idx)
        pp = B["per_park"][cpark]
        met = pp["metrics"]
        red = _red(cpark)
        order = ["HSKR", bn]
        colors = [TOK["primary"], "#E8505B"]

        # ── KPI
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3, gap="medium")
        k1.markdown(metric_card(f"+{red:.1f}%", "RMSE 감소율", delta=f"+vs {bn}"), unsafe_allow_html=True)
        k2.markdown(metric_card(f"{met['HSKR']['R2']:.3f}", "HSKR R²",
                                delta=f"+기존 {met[bn]['R2']:.3f}"), unsafe_allow_html=True)
        k3.markdown(metric_card(f"{met['HSKR']['RMSE']/1e4:.1f}만", "HSKR RMSE",
                                delta=f"+기존 {met[bn]['RMSE']/1e4:.1f}만"), unsafe_allow_html=True)

        # ── 시계열 (실제 vs HSKR vs 최고 기존)
        st.markdown('<h2 class="h-section" style="margin-top:24px">실제 vs 예측 (테스트 구간)</h2>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        if "dates_all" in pp and "y_all" in pp:
            fig.add_trace(go.Scatter(x=pp["dates_all"], y=pp["y_all"], name="실제(전체)", mode="lines",
                                     line=dict(color="rgba(140,140,150,0.30)", width=1.3)))
        fig.add_trace(go.Scatter(x=pp["dates_test"], y=pp["y_test"], name="실제", mode="lines+markers",
                                 line=dict(color=TOK["ink"], width=2), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=pp["dates_test"], y=pp["base_pred_test"][bn], name=f"기존({bn})",
                                 mode="lines+markers", line=dict(color="#E8505B", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=pp["dates_test"], y=pp["hskr_pred_test"], name="HSKR",
                                 mode="lines+markers", line=dict(color=TOK["primary"], width=3)))
        fig.update_layout(title=f"{cpark} — 테스트 구간", height=430, hovermode="x unified", yaxis_title="총이용객")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── 모델 비교 (RMSE / R²)
        st.markdown('<h2 class="h-section" style="margin-top:20px">모델 비교 (HSKR vs Ridge)</h2>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2, gap="medium")
        with ca:
            fr = go.Figure(go.Bar(x=[met[m]["RMSE"] / 1e4 for m in order], y=order, orientation="h",
                                  marker_color=colors, text=[f"{met[m]['RMSE']/1e4:.1f}" for m in order],
                                  textposition="outside"))
            fr.update_layout(title="RMSE (낮을수록 우수)", height=340, xaxis_title="RMSE(만)",
                             yaxis=dict(autorange="reversed"))
            style_fig(fr)
            st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})
        with cb:
            fR = go.Figure(go.Bar(x=[met[m]["R2"] for m in order], y=order, orientation="h",
                                  marker_color=colors, text=[f"{met[m]['R2']:.3f}" for m in order],
                                  textposition="outside"))
            fR.update_layout(title="R² (높을수록 우수)", height=340, xaxis_title="R²",
                             yaxis=dict(autorange="reversed"))
            style_fig(fR)
            st.plotly_chart(fR, use_container_width=True, config={"displayModeBar": False})

        # ── 실제 vs HSKR 산점도 + 공원별 향상률
        cc, cd = st.columns(2, gap="medium")
        with cc:
            yt, yp = np.asarray(pp["y_test"], float), np.asarray(pp["hskr_pred_test"], float)
            lim = [float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))]
            fsc = go.Figure()
            fsc.add_trace(go.Scatter(x=yt, y=yp, mode="markers",
                                     marker=dict(color=TOK["primary"], size=9, opacity=0.75), name="HSKR"))
            fsc.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color=TOK["ink"], dash="dot"), showlegend=False))
            fsc.update_layout(title="실제 vs HSKR 예측", height=380,
                              xaxis_title="실제", yaxis_title="예측")
            style_fig(fsc)
            st.plotly_chart(fsc, use_container_width=True, config={"displayModeBar": False})
        with cd:
            reds = sorted(((p, _red(p)) for p in parks), key=lambda x: x[1])
            barc = ["#E8505B" if p in core else TOK["primary"] for p, _ in reds]
            fp = go.Figure(go.Bar(x=[v for _, v in reds], y=[p.replace("한강공원", "") for p, _ in reds],
                                  orientation="h", marker_color=barc,
                                  text=[f"{v:+.0f}%" for _, v in reds], textposition="outside"))
            fp.add_vline(x=0, line=dict(color=TOK["ink"], dash="dot"))
            fp.update_layout(title="공원별 RMSE 감소율 vs Ridge (빨강=핵심3)", height=380, xaxis_title="감소율 %")
            style_fig(fp)
            st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})

        # ── 지표 표 + 요약
        st.markdown('<h2 class="h-section" style="margin-top:12px">성능 지표 표</h2>', unsafe_allow_html=True)
        mdf = pd.DataFrame([{"모델": m, "R²": round(met[m]["R2"], 3),
                             "RMSE(만)": round(met[m]["RMSE"] / 1e4, 1)} for m in order])
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        if core:
            avg_red = float(np.mean([_red(p) for p in core]))
            st.markdown(f'<div class="caption" style="margin-top:8px">핵심 3개 공원'
                        f'({" · ".join(p.replace("한강공원","") for p in core)}) 평균 RMSE 감소율 '
                        f'<b style="color:var(--primary)">+{avg_red:.1f}%</b> (HSKR vs Ridge).</div>',
                        unsafe_allow_html=True)
        st.caption("※ 전 11개 공원을 공원별로 학습(시계열 holdout). 대형·구조변화 공원(뚝섬·여의도 등)은 "
                   "예측이 어려워 R²가 음수일 수 있으나 HSKR이 Ridge 대비 RMSE를 줄입니다. "
                   "Ridge는 동일 테스트 구간의 기존 Ridge 베이스라인.")
    tile_close()


elif page == "핵심 변수 선별 효과":
    tile_open("light", anchor="featimp")
    st.markdown('<h2 class="h-display" style="color:var(--ink)">비교 2 — VIF + Feature Importance</h2>',
                unsafe_allow_html=True)
    st.markdown('<p class="lead">비교 1의 1등 모델 하나로, <b>VIF 피처</b> vs <b>중요도 상위 피처</b>의 예측 성능을 '
                '정량 비교하고 Q-Q·잔차·SHAP·Force·LIME·변수 중요도로 해석합니다.</p>', unsafe_allow_html=True)
    tile_close()

    FB, fi_err = load_fi()
    tile_open("light")
    if FB is None:
        st.markdown(f'<div class="card"><div class="body-strong">분석 결과(fi_models.pkl)를 불러올 수 없습니다</div>'
                    f'<div class="caption" style="margin-top:8px">model/fi_models.pkl 필요.<br/>'
                    f'진단: <code>{fi_err}</code></div></div>', unsafe_allow_html=True)
    else:
        try:
            from scipy.stats import probplot
            import shap
            import matplotlib.pyplot as plt

            mf, mt = FB["metrics_full"], FB["metrics_top"]
            allf, topf = list(FB["all_features"]), list(FB["top_features"])
            importance = FB.get("importance", {})
            std = [m for m in mf if m != "HSKR"]
            best = max(std, key=lambda m: mf[m]["R2"]) if std else list(mf)[0]
            kind = "linear" if best in ("Ridge", "ElasticNet") else "tree"
            mvb, mtb = mf[best], mt[best]      # 업로드 pkl 기준 best 모델 full vs top
            nv, ni = len(allf), len(topf)

            @st.cache_resource(show_spinner=f"{best} 진단 모델 학습·SHAP 계산 중...")
            def _diag(_pm, best, cols):
                from sklearn.base import clone
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import KFold, cross_val_predict
                from sklearn.linear_model import Ridge, ElasticNet
                from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                                              ExtraTreesRegressor)
                FAC = {"Ridge": Ridge(alpha=10.0),
                       "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000),
                       "GradientBoosting": GradientBoostingRegressor(random_state=42),
                       "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
                       "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)}
                cols = [c for c in cols if c in _pm.columns]
                X = _pm[cols].astype(float).fillna(0).values
                y = _pm["총이용객"].values.astype(float)
                Z = StandardScaler().fit_transform(X)
                est = FAC.get(best, Ridge(alpha=10.0))
                mdl = clone(est).fit(Z, y)
                oof = cross_val_predict(clone(est), Z, y, cv=KFold(5, shuffle=True, random_state=42))
                if best in ("Ridge", "ElasticNet"):
                    ex = shap.LinearExplainer(mdl, Z)
                else:
                    ex = shap.TreeExplainer(mdl)
                sv = np.asarray(ex.shap_values(Z))
                base_val = float(np.ravel(ex.expected_value)[0])
                return dict(Z=Z, y=y, oof=oof, mdl=mdl, sv=sv, base_val=base_val,
                            cols=[c for c in cols if c in _pm.columns])

            D = _diag(pm, best, tuple(topf))
            Z, y, oof, mdl, sv, base_val = D["Z"], D["y"], D["oof"], D["mdl"], D["sv"], D["base_val"]
            diag_cols = D["cols"]

            # ── 비교2: VIF vs VIF+중요도 (업로드 pkl 수치) — 전 모델 + best 강조
            st.markdown('<h2 class="h-section">실험 결과 — VIF vs VIF+중요도 (업로드 분석 결과)</h2>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="caption" style="margin-bottom:8px;line-height:1.6">업로드한 <b>fi_models.pkl</b> '
                        f'기준. VIF 적용 <b>{nv}개</b> 피처 vs 중요도 상위 <b>{ni}개</b>로 줄였을 때의 성능. '
                        f'표준 ML 1등은 <b style="color:var(--primary)">{best}</b>(HSKR은 신규 모델, 별도). '
                        f'변수를 {nv}→{ni}개로 줄여도 성능이 유지되는지 봅니다.</div>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3, gap="medium")
            k1.markdown(metric_card(f"{nv}→{ni}", "피처 수 (VIF→VIF+중요도)"), unsafe_allow_html=True)
            k2.markdown(metric_card(f"{mtb['R2']:.3f}", f"{best} · VIF+중요도 R²",
                                    delta=f"{mtb['R2']-mvb['R2']:+.3f} vs VIF {mvb['R2']:.3f}"), unsafe_allow_html=True)
            k3.markdown(metric_card(f"{mtb['RMSE']/1e4:.1f}만", f"{best} · VIF+중요도 RMSE",
                                    delta=f"{(mtb['RMSE']-mvb['RMSE'])/1e4:+.1f}만 vs VIF"), unsafe_allow_html=True)

            exp_df = pd.DataFrame([
                {"구분": f"VIF ({nv}개)", "R²": round(mvb["R2"], 4), "RMSE(만)": round(mvb["RMSE"] / 1e4, 1)},
                {"구분": f"VIF+중요도 ({ni}개)", "R²": round(mtb["R2"], 4), "RMSE(만)": round(mtb["RMSE"] / 1e4, 1)},
            ])
            ca, cb = st.columns([2, 3], gap="medium")
            ca.dataframe(exp_df, use_container_width=True, hide_index=True)
            with cb:
                fx = go.Figure(go.Bar(x=[f"VIF({nv})", f"VIF+중요도({ni})"], y=[mvb["R2"], mtb["R2"]],
                                      marker_color=TOK["primary"],
                                      text=[f"{mvb['R2']:.3f}", f"{mtb['R2']:.3f}"], textposition="outside"))
                fx.update_layout(title=f"{best} R² — VIF vs VIF+중요도", height=320, yaxis_title="R²")
                style_fig(fx)
                st.plotly_chart(fx, use_container_width=True, config={"displayModeBar": False})

            # ── 변수 중요도 (업로드 importance, VIF 24개) — 검색량 빨강 + 선택 top 강조
            st.markdown('<h2 class="h-section" style="margin-top:24px">변수 중요도 — 상위 '
                        f'{ni}개 선택</h2>', unsafe_allow_html=True)
            st.markdown('<div class="caption" style="margin-bottom:8px"><b style="color:#E8505B">빨강 = 검색량</b> · '
                        f'<b style="color:var(--primary)">진한 파랑 = 선택된 상위 {ni}개</b> · 연회색 = 미선택 '
                        '(업로드 fi_models.pkl importance)</div>', unsafe_allow_html=True)
            if importance:
                imp_df = pd.DataFrame({"변수": list(importance.keys()),
                                       "중요도": list(importance.values())}).sort_values("중요도")
                sel = set(topf)

                def _c(v):
                    return "#E8505B" if v == "검색량" else (TOK["primary"] if v in sel else TOK["ink_48"])
                fi = go.Figure(go.Bar(x=imp_df["중요도"], y=imp_df["변수"], orientation="h",
                                      marker_color=[_c(v) for v in imp_df["변수"]],
                                      text=[f"{v:.3f}" for v in imp_df["중요도"]], textposition="outside"))
                fi.update_layout(title=f"변수 중요도 (VIF {nv}개 중 · 빨강=검색량, 파랑=선택 {ni})",
                                 height=max(380, 24 * len(imp_df)), xaxis_title="중요도", margin=dict(l=10, r=60))
                style_fig(fi)
                st.plotly_chart(fi, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("importance 정보가 fi_models.pkl에 없습니다.")

            # ── 잔차 진단: Residual + Q-Q (VIF+중요도 모델 OOF)
            resid = y - oof
            st.markdown(f'<h2 class="h-section" style="margin-top:24px">잔차 진단 ({best} · VIF+중요도 {ni}개 · OOF)</h2>',
                        unsafe_allow_html=True)
            d1, d2 = st.columns(2, gap="medium")
            with d1:
                fr = go.Figure(go.Scatter(x=oof, y=resid, mode="markers",
                                          marker=dict(color=TOK["primary"], size=6, opacity=0.6)))
                fr.add_hline(y=0, line=dict(color=TOK["ink"], dash="dot"))
                fr.update_layout(title="잔차 vs 예측값", xaxis_title="예측", yaxis_title="잔차", height=380)
                style_fig(fr)
                st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})
            with d2:
                (osm, osr), _ = probplot(resid, dist="norm")
                fq = go.Figure()
                fq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=TOK["primary"], size=6)))
                fq.add_trace(go.Scatter(x=osm, y=osm * np.std(resid), mode="lines",
                                        line=dict(color=TOK["ink"], dash="dot")))
                fq.update_layout(title="Q-Q Plot (잔차 정규성)", showlegend=False, height=380)
                style_fig(fq)
                st.plotly_chart(fq, use_container_width=True, config={"displayModeBar": False})

            # ── SHAP Summary
            exp_name = "Linear" if kind == "linear" else "Tree"
            st.markdown(f'<h2 class="h-section" style="margin-top:22px">SHAP 해석 ({exp_name}Explainer)</h2>',
                        unsafe_allow_html=True)
            st.markdown('<div class="caption" style="margin-bottom:6px">각 점 = 한 예측 · 가로축 = SHAP value '
                        '(예측 기여) · 색 = 표준화 변수값(빨강 높음 / 파랑 낮음).</div>', unsafe_allow_html=True)
            fig_s = plt.figure(figsize=(9, 5))
            shap.summary_plot(sv, features=Z, feature_names=diag_cols, max_display=len(diag_cols), show=False)
            plt.title(f"SHAP Summary ({best} · VIF+중요도)", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_s, clear_figure=True)

            # ── 개별 예측: Force + LIME
            st.markdown('<h2 class="h-section" style="margin-top:26px">개별 예측 — Force plot · LIME</h2>',
                        unsafe_allow_html=True)
            ym = pd.to_datetime(pm["연월"].values)
            labels = [f"{i:>3} · {pm.iloc[i]['공원명']} · {pd.Timestamp(ym[i]):%Y-%m}" for i in range(len(pm))]
            default_i = next((i for i in range(len(pm)) if pm.iloc[i]["공원명"] == selected_park), 0)
            idx = st.selectbox("설명할 예측 (기본 = 선택 공원)", range(len(pm)), index=default_i,
                               format_func=lambda i: labels[i])
            pred_i = base_val + sv[idx, :].sum()
            f1, f2 = st.columns(2, gap="medium")
            f1.markdown(metric_card(f"{pred_i/1e4:,.1f}만", f"예측 — {pm.iloc[idx]['공원명']}"), unsafe_allow_html=True)
            f2.markdown(metric_card(f"{base_val/1e4:,.1f}만", "기준값(평균 예측)"), unsafe_allow_html=True)
            st.markdown('<div class="caption" style="margin:6px 0 4px 0">기준값에서 '
                        '<b style="color:#ff0d57">빨강(↑)</b> · <b style="color:#1e88e5">파랑(↓)</b> '
                        '변수 기여를 거쳐 최종 예측에 도달합니다. (변수값은 표준화 z-score)</div>', unsafe_allow_html=True)
            shap.force_plot(base_val, sv[idx, :], features=np.round(Z[idx, :], 2), feature_names=diag_cols,
                            matplotlib=True, show=False, figsize=(20, 3), text_rotation=12)
            st.pyplot(plt.gcf(), clear_figure=True)

            st.markdown('<div class="body-strong" style="margin-top:18px">LIME — 국소 선형 근사 (경량 자체구현)</div>',
                        unsafe_allow_html=True)
            lime_df = lime_explain(mdl.predict, Z[idx], Z, diag_cols, n_top=min(12, len(diag_cols)))
            colL = ["#E8505B" if v < 0 else TOK["primary"] for v in lime_df["국소기여"]]
            fl = go.Figure(go.Bar(x=lime_df["국소기여"], y=lime_df["변수"], orientation="h", marker_color=colL,
                                  text=[f"{v/1e3:+.1f}K" for v in lime_df["국소기여"]], textposition="outside"))
            fl.add_vline(x=0, line=dict(color=TOK["ink"], dash="dot"))
            fl.update_layout(title=f"LIME 국소 기여 — {pm.iloc[idx]['공원명']} {pd.Timestamp(ym[idx]):%Y-%m}",
                             height=380, xaxis_title="국소 기여 (파랑 ↑ / 빨강 ↓)", margin=dict(l=10, r=70))
            style_fig(fl)
            st.plotly_chart(fl, use_container_width=True, config={"displayModeBar": False})
            st.caption("※ 경량 LIME = 표본 주변 perturbation의 거리가중 Ridge 국소근사(추가 의존성 0). "
                       "설명력이 부족하면 아래 CSV(또는 SHAP 값)를 내려받아 lime/shap 패키지로 정밀 분석하세요.")

            # ── 다운로드
            st.markdown('<h2 class="h-section" style="margin-top:18px">다운로드</h2>', unsafe_allow_html=True)
            shap_imp = (pd.DataFrame({"변수": diag_cols, "평균_abs_SHAP": np.abs(sv).mean(0)})
                        .sort_values("평균_abs_SHAP", ascending=False))
            dd = st.columns(3, gap="medium")
            dd[0].download_button("⬇️ VIF/중요도 실험표 CSV", exp_df.to_csv(index=False).encode("utf-8-sig"),
                                  "vif_importance_experiment.csv", "text/csv", use_container_width=True)
            dd[1].download_button("⬇️ SHAP 평균기여 CSV", shap_imp.to_csv(index=False).encode("utf-8-sig"),
                                  "shap_importance.csv", "text/csv", use_container_width=True)
            dd[2].download_button("⬇️ LIME 기여 CSV", lime_df.to_csv(index=False).encode("utf-8-sig"),
                                  "lime_local.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.markdown(f'<div class="caption">분석을 실행할 수 없습니다 — {type(e).__name__}: {e}</div>',
                        unsafe_allow_html=True)
    tile_close()


elif page == "예측 시뮬레이터":
    tile_open("light", anchor="simulator")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--ink)">입력을 바꾸면, 예측이 즉시.</h2>
    <p class="lead">공원과 시설·검색량을 조정하면 RandomForest가 월 이용객을 다시 추정합니다.</p>
    """, unsafe_allow_html=True)
    tile_close()

    tile_open("light")
    model = bundle["model"]
    sim_cols = [c for c in feature_cols if c not in ("월sin", "월cos")]  # 계절성은 월 선택으로 처리

    st.markdown('<h2 class="h-section">입력 값 조정</h2>', unsafe_allow_html=True)
    top = st.columns(2, gap="large")
    sim_park = top[0].selectbox("공원", park_list,
                                index=(park_list.index(selected_park) if selected_park in park_list else 0))
    sim_month = top[1].slider("월", 1, 12, 6)
    st.markdown('<div class="caption" style="margin:6px 0 16px 0">시설/검색량 기본값은 해당 공원의 중앙값입니다.</div>',
                unsafe_allow_html=True)

    park_rows = pm[pm["공원명"] == sim_park]
    scols = st.columns(2, gap="large")
    vals = {}
    for i, c in enumerate(sim_cols):
        s = park_rows[c] if len(park_rows) else pm[c]
        lo, hi, med = float(s.min()), float(s.max()), float(s.median())
        if hi <= lo:
            hi = lo + 1.0
        step = max((hi - lo) / 100.0, 0.1)
        with scols[i % 2]:
            vals[c] = st.slider(c, lo, hi, med, step=step)

    row = dict(vals)
    row["월sin"] = float(np.sin(2 * np.pi * sim_month / 12))
    row["월cos"] = float(np.cos(2 * np.pi * sim_month / 12))
    row["공원명"] = sim_park
    Xrow = pd.DataFrame([[row[c] for c in feature_cols] + [sim_park]], columns=feature_cols + ["공원명"])

    try:
        pred = float(model.predict(Xrow)[0])
    except Exception as e:
        pred = None
        st.markdown(f'<div class="caption">예측 오류 — {e}</div>', unsafe_allow_html=True)

    if pred is not None:
        avg = float(park_rows["총이용객"].mean()) if len(park_rows) else float(pm["총이용객"].mean())
        diff = (pred - avg) / avg * 100 if avg else 0.0
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2, gap="medium")
        r1.markdown(metric_card(f"{pred/1e4:,.1f}만 명", f"예측 이용객 ({sim_park} · {sim_month}월)",
                                delta=f"{'+' if diff >= 0 else ''}{diff:.1f}% vs 공원 평균"),
                    unsafe_allow_html=True)
        r2.markdown(metric_card(f"{avg/1e4:,.1f}만 명", f"{sim_park} 평균 (실측)"),
                    unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=["예측값", "공원 평균"], y=[pred, avg],
            marker_color=[TOK["primary"], TOK["ink_48"]],
            text=[f"{pred/1e4:.1f}만", f"{avg/1e4:.1f}만"], textposition="outside", width=[0.5, 0.5]))
        fig.update_layout(title="예측 vs 공원 실측 평균", height=360, yaxis_title="월 이용객 수")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tile_close()


elif page == "Conformal":
    tile_open("light", anchor="uncertainty")
    st.markdown(f"""
    <h2 class="h-display" style="color:var(--ink)">분포 가정 없는 예측 구간.</h2>
    <p class="lead">Split Conformal로 보장된 커버리지를 — 단 하나의 캘리브레이션 단계로.</p>
    """, unsafe_allow_html=True)
    tile_close()

    tile_open("light")
    st.markdown('<h2 class="h-section">Conformal Prediction</h2>', unsafe_allow_html=True)

    try:
        from sklearn.model_selection import train_test_split

        alpha = st.slider("유의수준 α", 0.05, 0.30, 0.10, 0.05,
                          help="1-α 가 커버리지 (예: α=0.10 → 90% 구간)")

        model = bundle["model"]
        # 학습에 쓰지 않은 홀드아웃을 보정/검정으로 분할 (split conformal)
        X_cal, X_tst, y_cal, y_tst = train_test_split(bundle["Xte"], bundle["yte"],
                                                      test_size=0.5, random_state=1)
        cal_pred = model.predict(X_cal)
        q = float(np.quantile(np.abs(y_cal - cal_pred), 1 - alpha))
        pred = model.predict(X_tst)
        lower, upper = pred - q, pred + q
        coverage = float(np.mean((y_tst >= lower) & (y_tst <= upper))) * 100

        c1, c2 = st.columns([1, 3], gap="medium")
        c1.markdown(metric_card(f"{coverage:.1f}%", "실측 커버리지", delta=f"+목표 {(1-alpha)*100:.0f}%"),
                    unsafe_allow_html=True)

        with c2:
            order = np.argsort(y_tst)
            xi = np.arange(len(y_tst))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xi, y=upper[order], mode="lines",
                                     line=dict(color="rgba(0,102,204,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=xi, y=lower[order], mode="lines",
                                     line=dict(color="rgba(0,102,204,0)"),
                                     fill="tonexty", fillcolor="rgba(0,102,204,0.18)",
                                     name=f"{int((1-alpha)*100)}% 구간"))
            fig.add_trace(go.Scatter(x=xi, y=pred[order], mode="lines",
                                     name="예측", line=dict(color=TOK["primary"], width=2.0)))
            fig.add_trace(go.Scatter(x=xi, y=np.asarray(y_tst)[order], mode="markers",
                                     name="실측", marker=dict(color=TOK["ink"], size=5)))
            fig.update_layout(title="예측 구간 (테스트셋, 실측 오름차순)", xaxis_title="표본", yaxis_title="이용객")
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">Conformal 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "Bootstrap CI":
    tile_open("parchment", anchor="bootstrap")
    st.markdown('<h2 class="h-section">Bootstrap 95% 신뢰구간</h2>', unsafe_allow_html=True)

    try:
        from sklearn.metrics import r2_score, mean_absolute_error

        n_boot = st.slider("Bootstrap 반복 수", 100, 2000, 500, 100)

        yte = np.asarray(bundle["y"], dtype=float)
        pred = np.asarray(bundle["oof"], dtype=float)   # 폴드별 검증 예측 (전 815건)
        n = len(yte)
        rng = np.random.default_rng(7)
        r2s, maes = [], []
        for _ in range(n_boot):
            s = rng.integers(0, n, n)
            r2s.append(r2_score(yte[s], pred[s]))
            maes.append(mean_absolute_error(yte[s], pred[s]))
        r2s, maes = np.array(r2s), np.array(maes)

        c1, c2 = st.columns(2, gap="medium")
        c1.markdown(metric_card(f"{r2s.mean():.3f}", "Test R² (평균)",
                                delta=f"+95% CI [{np.percentile(r2s,2.5):.3f}, {np.percentile(r2s,97.5):.3f}]"),
                    unsafe_allow_html=True)
        c2.markdown(metric_card(f"{maes.mean()/1000:.1f}K", "Test MAE (평균)",
                                delta=f"+95% CI [{np.percentile(maes,2.5)/1000:.1f}K, {np.percentile(maes,97.5)/1000:.1f}K]"),
                    unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=r2s, nbinsx=40, marker_color=TOK["primary"]))
        fig.add_vline(x=np.percentile(r2s, 2.5), line=dict(color=TOK["ink"], dash="dot"))
        fig.add_vline(x=np.percentile(r2s, 97.5), line=dict(color=TOK["ink"], dash="dot"))
        fig.update_layout(title=f"Bootstrap R² 분포 (n={n_boot}, 95% CI 점선)", height=420,
                          xaxis_title="R²", yaxis_title="빈도")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.markdown(f'<div class="caption">Bootstrap 오류 — {e}</div>', unsafe_allow_html=True)
    tile_close()


elif page == "Nested CV":
    tile_open("light", anchor="nested-cv")
    st.markdown('<h2 class="h-section">Nested Cross-Validation</h2>', unsafe_allow_html=True)

    st.markdown('<div class="caption" style="margin-bottom:14px">'
                'RandomForest의 max_depth를 내부 루프에서 튜닝하고, 외부 루프로 일반화 성능을 추정합니다. '
                '(공원-월 815건)</div>', unsafe_allow_html=True)
    outer_k = st.slider("Outer fold", 3, 7, 5)
    inner_k = st.slider("Inner fold", 2, 5, 3)

    if st.button("Nested CV 실행", type="primary"):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import KFold, GridSearchCV
            from sklearn.metrics import r2_score

            X, y = bundle["X"], bundle["y"]

            def make_pipe():
                pre = ColumnTransformer([
                    ("num", "passthrough", feature_cols),
                    ("park", OneHotEncoder(handle_unknown="ignore"), ["공원명"])])
                return Pipeline([("pre", pre),
                                 ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])

            grid = {"rf__max_depth": [None, 10, 20]}
            outer = KFold(n_splits=outer_k, shuffle=True, random_state=42)
            scores, chosen = [], []
            with st.spinner(f"Nested CV 실행 중... (외부 {outer_k} × 내부 {inner_k})"):
                for tr, te in outer.split(X):
                    gs = GridSearchCV(make_pipe(), grid, cv=inner_k, scoring="r2", n_jobs=-1)
                    gs.fit(X.iloc[tr], y[tr])
                    scores.append(r2_score(y[te], gs.predict(X.iloc[te])))
                    chosen.append(gs.best_params_["rf__max_depth"])
            scores = np.array(scores)

            c1, c2, c3 = st.columns(3, gap="medium")
            c1.markdown(metric_card(f"{scores.mean():.3f}", "평균 R² (외부)"), unsafe_allow_html=True)
            c2.markdown(metric_card(f"±{scores.std():.3f}", "표준편차"), unsafe_allow_html=True)
            common = max(set(chosen), key=chosen.count)
            c3.markdown(metric_card(str(common), "선택 max_depth (최빈)"), unsafe_allow_html=True)

            st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
            cv_df = pd.DataFrame({"Fold": list(range(1, outer_k + 1)),
                                  "R²": np.round(scores, 4),
                                  "선택 depth": [str(c) for c in chosen]})
            fig = px.bar(cv_df, x="Fold", y="R²", text="선택 depth")
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
      <a href="#hskr">신규 모델 (HSKR)</a>
    </div>
    <div>
      <h5>해석</h5>
      <a href="#featimp">핵심 변수 선별 효과</a>
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
