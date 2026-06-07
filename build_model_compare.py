"""모델 예측 페이지용 Learning Curve + Q-Q(OOF 잔차) 사전계산 → model/model_compare.pkl.

성능 수치(R²·RMSE)는 사용자가 올린 model/fi_models.pkl(metrics_full)을 앱에서 직접 읽는다.
fi_models.pkl엔 LC·OOF가 없어, 같은 5개 모델을 **models_top에 저장된 사용자 하이퍼파라미터 그대로**
(clone) VIF 피처(all_features 24개)로 재학습해 learning curve와 OOF 예측(Q-Q용)만 계산한다.
새 모델 추가 없음 — 알고리즘·파라미터 모두 사용자 pkl 것.
실행: python3 build_model_compare.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, learning_curve, cross_val_predict
from sklearn.metrics import r2_score
from hskr_model import HybridSeasonalKernelRidge  # fi_models 언피클용

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
    mt = fi["models_top"]                       # 사용자 적합 모델(파라미터 추출용)
    models = [m for m in ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest", "ExtraTrees")
              if m in mt] + [m for m in mt if m not in
                             ("Ridge", "ElasticNet", "GradientBoosting", "RandomForest", "ExtraTrees", "HSKR")]
    X = df[feats].astype(float).fillna(0)
    print(f"LC/Q-Q 사전계산: {len(models)}개 모델, all_features {len(feats)}개, 표본 {len(y)}")

    kf = KFold(5, shuffle=True, random_state=42)
    lc, oof = {}, {}
    for m in models:
        est = clone(mt[m])                      # 사용자 하이퍼파라미터 그대로(미적합 복제)
        pipe = Pipeline([("sc", StandardScaler()), ("e", est)])
        oof[m] = cross_val_predict(pipe, X, y, cv=kf)
        ts, tr_s, va_s = learning_curve(Pipeline([("sc", StandardScaler()), ("e", clone(mt[m]))]), X, y,
                                        train_sizes=np.linspace(0.2, 1.0, 5),
                                        cv=KFold(4, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
        lc[m] = {"train_sizes": ts, "train_scores": tr_s, "val_scores": va_s}
        print(f"  {m:16s} OOF R²≈{r2_score(y, oof[m]):.3f}  LC최종검증≈{va_s.mean(1)[-1]:.3f}")

    out = {"models": models, "lc": lc, "oof": {m: np.asarray(v) for m, v in oof.items()},
           "y": y, "feats": feats, "n_samples": int(len(y))}
    p = os.path.join(BASE, "model", "model_compare.pkl")
    with open(p, "wb") as f:
        pickle.dump(out, f)
    print(f"\n저장: {p} (LC + OOF, 순수 숫자)")


if __name__ == "__main__":
    main()
