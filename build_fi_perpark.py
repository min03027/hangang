"""한강 대시보드 '비교1(모델 예측)·비교2(핵심 변수 선별 효과)' 공원별 번들 생성기.

→ 출력: model/fi_perpark.pkl  (Streamlit 앱이 공원별로 그대로 읽어 표시. 앱은 재학습 안 함)

설계: 학습된 sklearn 모델을 pkl에 넣지 않는다(버전-피클 위험 회피 + 용량 절감).
  대신 Colab에서 '내 모델'로 학습·SHAP·LIME까지 모두 계산해 **숫자 배열만** 저장한다.
  → pkl 안에는 numpy 배열·리스트·dict만 들어가므로 버전에 안전하고 작다(수백 KB~수 MB).

[버전] Colab 첫 셀:
  !pip -q install "numpy==2.4.6" "scikit-learn==1.8.0" "statsmodels==0.14.6" \
                  "pandas==3.0.3" "shap==0.52.0" openpyxl

입력(대시보드와 동일): users.csv, trend.xlsx  (Colab에 업로드, 또는 data/ 아래)
실행:  python build_fi_perpark.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

VIF_THRESH = 10.0     # VIF가 이 값보다 크면 가장 큰 변수부터 반복 제거
TOP_K = 6             # 중요도 상위 몇 개로 줄일지

# 대시보드 기존 fi_models.pkl의 VIF 후보 24개(이 풀에서 공원별로 다시 VIF 선정).
# None 으로 두면 데이터의 모든 수치형 시설 컬럼 + 검색량을 후보로 자동 사용.
CANDIDATE = ['자전거', '운동시설', '피아노물길', '계절,녹음수광장', '캠핑장', '천상계단',
             '너플들판테라스', '물빛광장', '검색량', '여의도샛강', '멀티프라자', '외국인',
             '마라톤', '세빛섬', '키즈랜드', '수상시설', '여의도시민 요트나루', '인라인',
             '음악분수', '서울색공원', '롤러장', '골프장', 'x게임장', '빙상장/눈설매장']


def make_models():
    """공원마다 다르게 튜닝하려면 이 함수만 바꾸면 됨."""
    return {
        "Ridge": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=1.0, max_iter=10000),
        "GradientBoosting": GradientBoostingRegressor(learning_rate=0.03,
                                                      min_samples_leaf=3, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    }


def _find(name):
    for p in (name, os.path.join("data", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} 을(를) 현재 폴더나 data/ 에서 찾을 수 없습니다.")


def build_pm():
    """대시보드와 동일하게 공원-월 단위 데이터프레임 생성(814행 = 11공원 × 74개월)."""
    df = pd.read_csv(_find("users.csv"))
    df["현황 일시"] = pd.to_datetime(df["현황 일시"])
    df["연월"] = df["현황 일시"].dt.to_period("M").dt.to_timestamp()
    tc = ["일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)"]
    for c in tc:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["총이용객"] = df[tc].sum(axis=1)
    tr = pd.read_excel(_find("trend.xlsx")).rename(columns={"날짜": "연월"})
    tr["연월"] = pd.to_datetime(tr["연월"])
    pcols = [c for c in tr.columns if c.endswith("한강공원") and c != "한강공원"]
    trl = tr.melt(id_vars="연월", value_vars=pcols, var_name="공원명", value_name="검색량")
    df = pd.merge(df, trl, on=["연월", "공원명"], how="left")
    df["검색량"] = pd.to_numeric(df["검색량"], errors="coerce")
    df["검색량"] = df["검색량"].fillna(df["검색량"].median())
    return df.drop_duplicates(subset=["공원명", "연월"], keep="first").reset_index(drop=True)


def vif_select(sub, pool):
    feats = list(pool)
    while len(feats) > 2:
        Z = StandardScaler().fit_transform(sub[feats].astype(float).fillna(0).values)
        vifs = [variance_inflation_factor(Z, i) for i in range(Z.shape[1])]
        j = int(np.argmax(vifs))
        if vifs[j] > VIF_THRESH:
            feats.pop(j)
        else:
            break
    return feats


def cv_metrics(sub, y, cols):
    """5-fold CV로 모델별 R²/RMSE + OOF 예측(잔차/Q-Q용)."""
    Z = StandardScaler().fit_transform(sub[cols].astype(float).fillna(0).values)
    kf = KFold(5, shuffle=True, random_state=42)
    out = {}
    for nm, est in make_models().items():
        oof = cross_val_predict(clone(est), Z, y, cv=kf)
        out[nm] = {"R2": float(r2_score(y, oof)),
                   "RMSE": float(mean_squared_error(y, oof) ** 0.5),
                   "oof": np.asarray(oof, float)}
    return out


def lime_local(predict_fn, Z, feat_names, n_samples=600, seed=0):
    """경량 LIME: 각 행 주변 perturbation의 거리가중 Ridge 국소근사 → 행×피처 기여 배열."""
    rng = np.random.RandomState(seed)
    n, d = Z.shape
    out = np.zeros((n, d), float)
    for i in range(n):
        P = Z[i] + rng.normal(0, 1.0, size=(n_samples, d))
        yp = predict_fn(P)
        w = np.exp(-(np.linalg.norm(P - Z[i], axis=1) ** 2) / (d))
        lr = Ridge(alpha=1.0).fit(P, yp, sample_weight=w)
        out[i] = lr.coef_ * Z[i]    # 국소 선형 기여(표준화 스케일)
    return out


def shap_diag(sub, y, cols, best):
    """best 모델 학습(표준화) → SHAP value·base·표준화값 Z + (예측함수, Z) 반환."""
    X = sub[cols].astype(float).fillna(0).values
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    mdl = clone(make_models()[best]).fit(Z, y)
    if best in ("Ridge", "ElasticNet"):
        ex = shap.LinearExplainer(mdl, Z)
    else:
        ex = shap.TreeExplainer(mdl)
    sv = np.asarray(ex.shap_values(Z), float)
    base = float(np.ravel(ex.expected_value)[0])
    return sv, base, Z, mdl


def main():
    df = build_pm()
    cand = CANDIDATE or [c for c in df.columns
                         if c not in ("공원명", "연월", "현황 일시", "총이용객",
                                      "일반이용자(아침)", "일반이용자(낮)", "일반이용자(저녁)")
                         and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in cand if c in df.columns]
    per_park = {}
    for park, sub in df.groupby("공원명"):
        sub = sub.sort_values("연월")
        y = sub["총이용객"].values.astype(float)
        pool = [c for c in cand if sub[c].astype(float).std() > 1e-9]   # 그 공원 변동 피처
        vf = vif_select(sub, pool)                                      # 공원별 VIF 선정
        mfull = cv_metrics(sub, y, vf)
        best = max(mfull, key=lambda m: mfull[m]["R2"])                 # 그 공원 1등 모델
        # best 모델 중요도 → top-K
        Zi = StandardScaler().fit_transform(sub[vf].astype(float).fillna(0).values)
        mi = clone(make_models()[best]).fit(Zi, y)
        raw = (np.abs(mi.feature_importances_) if hasattr(mi, "feature_importances_")
               else np.abs(np.ravel(mi.coef_)))
        s = float(raw.sum()) or 1.0
        importance = {c: float(v) for c, v in sorted(zip(vf, raw / s), key=lambda x: -x[1])}
        topf = list(importance)[:min(TOP_K, len(vf))]
        mtop = cv_metrics(sub, y, topf)
        # SHAP (VIF 전체 / top-K) + LIME (top-K)
        sv_full, base_full, Z_full, _ = shap_diag(sub, y, vf, best)
        sv_top, base_top, Z_top, mdl_top = shap_diag(sub, y, topf, best)
        lime_top = lime_local(mdl_top.predict, Z_top, topf)
        # learning curve
        lc = {}
        for nm, est in make_models().items():
            try:
                ts, tr, va = learning_curve(
                    Pipeline([("sc", StandardScaler()), ("e", clone(est))]),
                    sub[vf].astype(float).fillna(0), y,
                    train_sizes=np.linspace(0.3, 1.0, 5),
                    cv=KFold(4, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
                lc[nm] = {"train_sizes": ts, "train_scores": tr, "val_scores": va}
            except Exception:
                pass
        per_park[park] = {
            "n": int(len(sub)), "pool": pool,
            "all_features": vf, "top_features": topf, "importance": importance, "best": best,
            "metrics_full": mfull, "metrics_top": mtop,
            "y": y, "dates": pd.to_datetime(sub["연월"]).dt.strftime("%Y-%m").tolist(),
            "sv_full": sv_full, "Z_full": Z_full, "base_full": base_full,
            "sv_top": sv_top, "Z_top": Z_top, "base_top": base_top,
            "lime_top": lime_top, "lc": lc,
        }
        print(f"{park:12s} pool {len(pool):2d}  VIF {len(vf):2d}  top {len(topf)}  "
              f"best {best:14s} R²={mfull[best]['R2']:+.3f}")

    bundle = {"parks": list(per_park.keys()), "candidate_pool": cand,
              "per_park": per_park, "config": {"vif_thresh": VIF_THRESH, "top_k": TOP_K}}
    os.makedirs("model", exist_ok=True)
    out = os.path.join("model", "fi_perpark.pkl")
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    chk = pickle.load(open(out, "rb"))
    p0 = chk["parks"][0]
    print(f"\n저장 완료: {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    print(f"공원 수: {len(chk['parks'])}  | 예시({p0}) keys: {sorted(chk['per_park'][p0])}")


if __name__ == "__main__":
    main()
