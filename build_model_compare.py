"""모델 예측 페이지용 model/model_compare.pkl 생성.

전 11공원 공원-월(815건)에서 5개 표준 모델의 글로벌 성능(5-fold CV R²/RMSE/MAE)과
learning curve를 사전 계산. HSKR 풀링 metric은 hskr_model.pkl(공원별)에서 읽어 비교 표에 합산.
피처/파이프라인은 appnew.py get_bundle 과 동일(공원 원핫 + 검색량 + 계절성 + 시설).
실행: python3 build_model_compare.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE = os.path.dirname(os.path.abspath(__file__))


def build_pm():
    df = pd.read_csv(os.path.join(BASE, "data", "users.csv"))
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M").dt.to_timestamp()
    df["월"] = df["연월"].dt.month
    time_cols = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
    META = ["일련번호", "코드", "주소", "시명", "구명", "등록", "수정", "일시"]
    EXCL = set(time_cols) | {"총이용객", "월", "공원명", "현황 일시", "연월", "계절", "검색량", "월sin", "월cos"}
    base_feats = [c for c in df.columns if c not in EXCL and not any(h in c for h in META)]
    for c in time_cols + base_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[time_cols].sum(axis=1)
    df["월sin"] = np.sin(2 * np.pi * df["월"] / 12)
    df["월cos"] = np.cos(2 * np.pi * df["월"] / 12)
    tr = pd.read_excel(os.path.join(BASE, "data", "trend.xlsx")).rename(columns={"날짜": "연월"})
    tr["연월"] = pd.to_datetime(tr["연월"])
    pcols = [c for c in tr.columns if c.endswith("한강공원") and c != "한강공원"]
    trl = tr.melt(id_vars="연월", value_vars=pcols, var_name="공원명", value_name="검색량")
    df = pd.merge(df, trl, on=["연월", "공원명"], how="left")
    df["검색량"] = pd.to_numeric(df["검색량"], errors="coerce")
    df["검색량"] = df["검색량"].fillna(df["검색량"].median())
    df = df.drop_duplicates(subset=["공원명", "연월"], keep="first").reset_index(drop=True)
    fac = [c for c in base_feats if df[c].std() > 0]
    feats = fac + ["검색량", "월sin", "월cos"]
    return df, feats


def main():
    df, feats = build_pm()
    X = df[feats + ["공원명"]].copy()
    y = df["총이용객"].values.astype(float)
    pre = lambda: ColumnTransformer([("num", "passthrough", feats),
                                     ("park", OneHotEncoder(handle_unknown="ignore"), ["공원명"])])
    models = {
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }
    kf = KFold(5, shuffle=True, random_state=42)
    metrics, lc = {}, {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre()), ("est", est)])
        oof = cross_val_predict(pipe, X, y, cv=kf)
        # 폴드별 R² 평균/표준편차
        r2s = []
        for tr_i, te_i in kf.split(X):
            r2s.append(r2_score(y[te_i], oof[te_i]))  # oof 기반 근사
        metrics[name] = {"R2": float(r2_score(y, oof)),
                         "R2_std": float(np.std(r2s)),
                         "RMSE": float(mean_squared_error(y, oof) ** 0.5),
                         "MAE": float(mean_absolute_error(y, oof))}
        ts, tr_sc, va_sc = learning_curve(Pipeline([("pre", pre()), ("est", est)]), X, y,
                                          train_sizes=np.linspace(0.2, 1.0, 5),
                                          cv=KFold(4, shuffle=True, random_state=42),
                                          scoring="r2", n_jobs=-1)
        lc[name] = {"train_sizes": ts, "train_scores": tr_sc, "val_scores": va_sc}
        print(f"  {name:16s} R²={metrics[name]['R2']:.3f}±{metrics[name]['R2_std']:.3f}  "
              f"RMSE={metrics[name]['RMSE']/1e4:.1f}만  MAE={metrics[name]['MAE']/1e3:.1f}K")

    # HSKR 풀링(공원별) metric 참고용
    hskr_pooled = None
    try:
        import sys; sys.path.insert(0, BASE)
        from hskr_model import HybridSeasonalKernelRidge  # noqa
        B = pickle.load(open(os.path.join(BASE, "model", "hskr_model.pkl"), "rb"))
        parks = list(B["parks"])
        yt = np.concatenate([np.asarray(B["per_park"][p]["y_test"], float) for p in parks])
        hk = np.concatenate([np.asarray(B["per_park"][p]["hskr_pred_test"], float) for p in parks])
        hskr_pooled = {"R2": float(r2_score(yt, hk)), "RMSE": float(mean_squared_error(yt, hk) ** 0.5),
                       "MAE": float(mean_absolute_error(yt, hk)),
                       "note": "공원별 모델 테스트 구간 풀링(스케일 혼합 → R² 낮음, RMSE 비교용)"}
        print(f"  HSKR(풀링)        R²={hskr_pooled['R2']:.3f}  RMSE={hskr_pooled['RMSE']/1e4:.1f}만")
    except Exception as e:
        print("  HSKR 풀링 생략:", e)

    out = {"models": list(models.keys()), "metrics": metrics, "lc": lc,
           "n_samples": int(len(y)), "n_features": len(feats), "hskr_pooled": hskr_pooled}
    p = os.path.join(BASE, "model", "model_compare.pkl")
    with open(p, "wb") as f:
        pickle.dump(out, f)
    print(f"\n저장: {p}  (모델 {len(models)} + HSKR 참고)")


if __name__ == "__main__":
    main()
