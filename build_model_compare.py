"""모델 예측 페이지의 learning curve 전용 사전계산 → model/model_compare.pkl.

성능 지표·중요도·피처셋은 사용자가 올린 model/fi_models.pkl을 그대로 사용한다(앱에서 직접 읽음).
fi_models.pkl엔 learning curve가 없어, 동일한 all_features(VIF 적용 24개)로 표준 ML 3종
(Ridge·RandomForest·GradientBoosting)의 학습곡선만 여기서 계산해 보강한다(참고용).
실행: python3 build_model_compare.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, learning_curve

BASE = os.path.dirname(os.path.abspath(__file__))


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
    return df.drop_duplicates(subset=["공원명", "연월"], keep="first").reset_index(drop=True)


def main():
    df = build_pm()
    y = df["총이용객"].values.astype(float)
    fi = pickle.load(open(os.path.join(BASE, "model", "fi_models.pkl"), "rb"))
    feats = [c for c in fi["all_features"] if c in df.columns]
    X = df[feats].astype(float).fillna(0)
    print(f"learning curve 계산: all_features {len(feats)}개, 표본 {len(y)}")

    M = {"Ridge": Ridge(alpha=10.0),
         "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
         "GradientBoosting": GradientBoostingRegressor(random_state=42)}
    lc = {}
    for n, e in M.items():
        ts, tr_s, va_s = learning_curve(Pipeline([("sc", StandardScaler()), ("e", e)]), X, y,
                                        train_sizes=np.linspace(0.2, 1.0, 5),
                                        cv=KFold(4, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
        lc[n] = {"train_sizes": ts, "train_scores": tr_s, "val_scores": va_s}
        print(f"  {n:16s} 최종 검증 R² ≈ {va_s.mean(1)[-1]:.3f}")

    out = {"lc": lc, "feats": feats, "n_samples": int(len(y))}
    p = os.path.join(BASE, "model", "model_compare.pkl")
    with open(p, "wb") as f:
        pickle.dump(out, f)
    print(f"\n저장(learning curve 전용): {p}")


if __name__ == "__main__":
    main()
