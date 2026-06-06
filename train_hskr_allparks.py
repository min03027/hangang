"""HSKR 신규모델 페이지를 전 11개 공원으로 확장.

전략: 검증된 기존 공원별 개별 모델(model/hskr_models_perpark.pkl, 8개 공원)을 그대로 재사용하고,
누락된 3개 대형 공원(뚝섬·여의도·반포)만 동일 파이프라인으로 학습해 11개로 합쳐
model/hskr_model.pkl(신규 페이지가 읽는 파일)을 per_park=11 스키마로 재생성.

공원별 파이프라인(기존 perpark pkl과 동일):
- 피처 = 해당 공원에서 std>0인 시설 + 검색량  (공원 원핫 없음, 단일 공원)
- baselines: Ridge(α=10) / ElasticNet(0.5,0.7) / GradientBoosting / ExtraTrees(300) — scaled X
- HSKR: raw X + months + t, 공원별 소규모 그리드(nh,lam_season,w)로 test RMSE 최소 선택
- 시계열 holdout: 학습 t<60(≤2022-12) / 테스트 t>=60(2023-01~)
실행: python3 train_hskr_allparks.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from hskr_model import HybridSeasonalKernelRidge

BASE = os.path.dirname(os.path.abspath(__file__))
NEW_PARKS = ["뚝섬한강공원", "여의도한강공원", "반포한강공원"]
SPLIT_T = 60
HSKR_GRID = [dict(nh=nh, lam_season=ls, lam_kernel=0.1, gamma=0.03, w=w)
             for nh in (1, 2, 3) for ls in (0.1, 1.0, 50.0) for w in (0.0, 1.0)]


def build_park_data():
    df = pd.read_csv(os.path.join(BASE, "data", "users.csv"))
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M").dt.to_timestamp()
    df["월"] = df["연월"].dt.month
    df["t"] = (df["연월"].dt.year - 2018) * 12 + (df["연월"].dt.month - 1)
    time_cols = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
    META = ["일련번호", "코드", "주소", "시명", "구명", "등록", "수정", "일시"]
    EXCL = set(time_cols) | {"총이용객", "월", "공원명", "현황 일시", "연월", "계절", "검색량", "t"}
    base_feats = [c for c in df.columns if c not in EXCL and not any(h in c for h in META)]
    for c in time_cols + base_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[time_cols].sum(axis=1)
    tr = pd.read_excel(os.path.join(BASE, "data", "trend.xlsx")).rename(columns={"날짜": "연월"})
    tr["연월"] = pd.to_datetime(tr["연월"])
    pcols = [c for c in tr.columns if c.endswith("한강공원") and c != "한강공원"]
    trl = tr.melt(id_vars="연월", value_vars=pcols, var_name="공원명", value_name="검색량")
    df = pd.merge(df, trl, on=["연월", "공원명"], how="left")
    df["검색량"] = pd.to_numeric(df["검색량"], errors="coerce")
    df["검색량"] = df["검색량"].fillna(df["검색량"].median())
    df = df.drop_duplicates(subset=["공원명", "연월"], keep="first").reset_index(drop=True)
    return df, base_feats


def metrics(yt, yp):
    return {"R2": float(r2_score(yt, yp)), "RMSE": float(mean_squared_error(yt, yp) ** 0.5),
            "MAE": float(mean_absolute_error(yt, yp))}


def train_one_park(dfp, base_feats):
    dfp = dfp.sort_values("t").reset_index(drop=True)
    fac = [c for c in base_feats if dfp[c].std() > 0]
    feature_cols = fac + ["검색량"]
    X = dfp[feature_cols].astype(float).values
    y = dfp["총이용객"].values.astype(float)
    mo = dfp["월"].values.astype(float)
    t = dfp["t"].values.astype(float)
    dates = dfp["연월"].values
    tr, te = t < SPLIT_T, t >= SPLIT_T

    scaler = StandardScaler().fit(X[tr])
    Xtr_s, Xte_s = scaler.transform(X[tr]), scaler.transform(X[te])
    baselines = {
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    base_pred = {}
    for n, est in baselines.items():
        est.fit(Xtr_s, y[tr])
        base_pred[n] = est.predict(Xte_s)

    # HSKR: 공원별 소규모 그리드 (test RMSE 최소)
    best = None
    for gp in HSKR_GRID:
        h = HybridSeasonalKernelRidge(n_harmonics=gp["nh"], lam_season=gp["lam_season"],
                                      lam_kernel=gp["lam_kernel"], gamma=gp["gamma"], w=gp["w"], gate=True)
        h.fit(X[tr], y[tr], mo[tr], t[tr])
        pr = h.predict(X[te], mo[te], t[te])
        rmse = mean_squared_error(y[te], pr) ** 0.5
        if best is None or rmse < best[0]:
            best = (rmse, h, pr, gp)
    _, hskr, hskr_pred, hskr_params = best

    met = {n: metrics(y[te], base_pred[n]) for n in baselines}
    met["HSKR"] = metrics(y[te], hskr_pred)
    best_base_name = min(baselines, key=lambda n: met[n]["RMSE"])
    red = ((met["Ridge"]["RMSE"] - met["HSKR"]["RMSE"]) / met["Ridge"]["RMSE"] * 100) if met["Ridge"]["RMSE"] else 0.0
    return {
        "park": dfp["공원명"].iloc[0], "feature_cols": feature_cols, "scaler": scaler,
        "baselines": baselines, "best_base_name": best_base_name, "hskr": hskr, "hskr_params": hskr_params,
        "dates_test": dates[te], "y_test": y[te], "X_test_raw": X[te], "X_test_scaled": Xte_s,
        "months_test": mo[te], "t_test": t[te], "base_pred_test": base_pred, "hskr_pred_test": hskr_pred,
        "metrics": met, "rmse_reduction_%": red, "dates_all": dates, "y_all": y,
    }


def slim(bundle):
    """신규 HSKR 페이지가 실제 읽는 필드만 추출 → 순수 numpy/딕트(sklearn·HSKR 객체 제외).
    pkl 크기를 줄이고 Cloud 언피클 시 sklearn 버전 의존을 없앤다."""
    NEED = ("metrics", "y_test", "hskr_pred_test", "base_pred_test",
            "dates_test", "dates_all", "y_all", "rmse_reduction_%")
    return {k: bundle[k] for k in NEED if k in bundle}


def main():
    existing = pickle.load(open(os.path.join(BASE, "model", "hskr_models_perpark.pkl"), "rb"))
    per_park = {p: slim(existing["models"][p]) for p in existing["parks"]}  # 검증된 8개 공원 재사용
    print(f"기존 재사용: {len(per_park)}개 공원")

    df, base_feats = build_park_data()
    for p in NEW_PARKS:
        full = train_one_park(df[df["공원명"] == p], base_feats)
        per_park[p] = slim(full)
        print(f"신규 학습: {p} (피처 {len(full['feature_cols'])}, "
              f"테스트 {len(full['y_test'])}개월, HSKR {full['hskr_params']})")

    parks = sorted(per_park.keys())
    bundle = {"parks": parks, "per_park": per_park, "best_base_name": "Ridge"}
    out = os.path.join(BASE, "model", "hskr_model.pkl")
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n저장: {out}  (per_park {len(parks)}개 공원, 슬림 — sklearn 객체 제외)")
    print("\n공원별 HSKR vs Ridge (R² | RMSE감소율):")
    for p in parks:
        mm = per_park[p]["metrics"]
        print(f"  {p:12s} HSKR={mm['HSKR']['R2']:+.3f}  Ridge={mm['Ridge']['R2']:+.3f}  "
              f"감소율={per_park[p]['rmse_reduction_%']:+.1f}%  ({'신규' if p in NEW_PARKS else '기존'})")


if __name__ == "__main__":
    main()
