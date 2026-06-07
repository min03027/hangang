"""모델 비교 사전계산 → model/model_compare.pkl (순수 숫자/배열만, sklearn 객체 미포함).

사용자 설계(2단 비교):
  · base 피처 = fi_models.pkl의 all_features(엄선 24개) + 검색량  (단일공원 47개 kitchen-sink 아님)
  · 비교1 [VIF 적용]: base에 Stepwise VIF(>10) 적용 → 생존 피처로 ML 5개(Ridge·ElasticNet·
    GradientBoosting·RandomForest·ExtraTrees) 5-fold CV 비교 → 1등 모델 선정 (+ learning curve)
  · 비교2 [VIF + Feature Importance]: 1등 모델의 중요도 상위 K개만 골라 VIF 대비 성능 비교
공원 원핫·계절성 미사용(사용자 fi_models 방법론과 동일).
실행: python3 build_model_compare.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = os.path.dirname(os.path.abspath(__file__))
TOP_K = 6  # VIF+중요도 단계에서 선택할 상위 피처 수 (fi_models top_features 수와 동일)


def build_pm():
    df = pd.read_csv(os.path.join(BASE, "data", "users.csv"))
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M").dt.to_timestamp()
    tc = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
    for c in tc:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[tc].sum(axis=1)
    tr = pd.read_excel(os.path.join(BASE, "data", "trend.xlsx")).rename(columns={"날짜": "연월"})
    tr["연월"] = pd.to_datetime(tr["연월"])
    pcols = [c for c in tr.columns if c.endswith("한강공원") and c != "한강공원"]
    trl = tr.melt(id_vars="연월", value_vars=pcols, var_name="공원명", value_name="검색량")
    df = pd.merge(df, trl, on=["연월", "공원명"], how="left")
    df["검색량"] = pd.to_numeric(df["검색량"], errors="coerce")
    df["검색량"] = df["검색량"].fillna(df["검색량"].median())
    df = df.drop_duplicates(subset=["공원명", "연월"], keep="first").reset_index(drop=True)
    return df


def models():
    return {
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }


def stepwise_vif(df, cols, thr=10.0):
    Z = pd.DataFrame(StandardScaler().fit_transform(df[cols].astype(float).fillna(0)), columns=cols)
    keep, dropped = list(cols), []
    while len(keep) > 2:
        v = [variance_inflation_factor(Z[keep].values, i) for i in range(len(keep))]
        j = int(np.argmax(v))
        if v[j] > thr:
            dropped.append((keep[j], round(float(v[j]), 1))); keep.pop(j)
        else:
            break
    max_left = max(variance_inflation_factor(Z[keep].values, i) for i in range(len(keep)))
    return keep, dropped, float(max_left)


def metr(df, cols, est, y, kf):
    pipe = Pipeline([("sc", StandardScaler()), ("e", clone(est))])
    oof = cross_val_predict(pipe, df[cols].astype(float).fillna(0), y, cv=kf)
    return {"R2": float(r2_score(y, oof)), "RMSE": float(mean_squared_error(y, oof) ** 0.5),
            "MAE": float(mean_absolute_error(y, oof)), "oof": oof}


def main():
    df = build_pm()
    y = df["총이용객"].values.astype(float)
    fi = pickle.load(open(os.path.join(BASE, "model", "fi_models.pkl"), "rb"))
    base = [c for c in fi["all_features"] if c in df.columns]
    print(f"base(fi_models all_features) = {len(base)}개")

    # 비교1: VIF 적용 → 5개 모델
    vif_keep, vif_dropped, max_left = stepwise_vif(df, base)
    print(f"VIF 적용: {len(base)}→{len(vif_keep)}개 (제거 {len(vif_dropped)}, 최대잔여 VIF {max_left:.1f})")
    kf = KFold(5, shuffle=True, random_state=42)
    M = models()
    metrics_vif, lc = {}, {}
    for n, e in M.items():
        m = metr(df, vif_keep, e, y, kf)
        r2s = [r2_score(y[te], m["oof"][te]) for _, te in kf.split(df)]
        metrics_vif[n] = {"R2": m["R2"], "R2_std": float(np.std(r2s)), "RMSE": m["RMSE"], "MAE": m["MAE"]}
        ts, tr_s, va_s = learning_curve(Pipeline([("sc", StandardScaler()), ("e", clone(e))]),
                                        df[vif_keep].astype(float).fillna(0), y,
                                        train_sizes=np.linspace(0.2, 1.0, 5),
                                        cv=KFold(4, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
        lc[n] = {"train_sizes": ts, "train_scores": tr_s, "val_scores": va_s}
        print(f"  {n:16s} R²={metrics_vif[n]['R2']:.3f}  RMSE={metrics_vif[n]['RMSE']/1e4:.1f}만")
    best = max(metrics_vif, key=lambda n: metrics_vif[n]["R2"])
    best_kind = "linear" if best in ("Ridge", "ElasticNet") else "tree"
    print(f"  >>> 비교1 1등: {best} ({best_kind})")

    # 비교2: 1등 모델 중요도 → 상위 K → VIF vs VIF+중요도
    Xv = StandardScaler().fit_transform(df[vif_keep].astype(float).fillna(0))
    bm = clone(M[best]).fit(Xv, y)
    imp_vals = bm.feature_importances_ if hasattr(bm, "feature_importances_") else np.abs(bm.coef_)
    importance = {vif_keep[i]: float(imp_vals[i]) for i in range(len(vif_keep))}
    imp_keep = [vif_keep[i] for i in np.argsort(imp_vals)[::-1][:TOP_K]]
    m_vif = metr(df, vif_keep, M[best], y, kf)
    m_imp = metr(df, imp_keep, M[best], y, kf)
    print(f"  비교2 {best}: VIF({len(vif_keep)}) R²={m_vif['R2']:.3f} → VIF+중요도({len(imp_keep)}) R²={m_imp['R2']:.3f}")
    print(f"  중요도 top-{TOP_K}: {imp_keep}")

    out = {
        "base_feats": base, "vif_keep": vif_keep, "vif_dropped": vif_dropped, "max_vif_left": max_left,
        "models": list(M.keys()), "metrics_vif": metrics_vif, "lc": lc,
        "best": best, "best_kind": best_kind,
        "importance": importance, "imp_keep": imp_keep, "top_k": TOP_K,
        "metrics_best_vif": {k: m_vif[k] for k in ("R2", "RMSE", "MAE")},
        "metrics_best_imp": {k: m_imp[k] for k in ("R2", "RMSE", "MAE")},
        "n_samples": int(len(y)),
    }
    p = os.path.join(BASE, "model", "model_compare.pkl")
    with open(p, "wb") as f:
        pickle.dump(out, f)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
