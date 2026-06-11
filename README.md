# 🏞️ 한강공원 이용객 분석 대시보드

서울시 한강공원 11개소의 월별 이용객 현황과 네이버 검색 트렌드를 분석·예측하는 Streamlit 대시보드입니다.

## 기능

- **EDA**: 계절·월별 이용 패턴, 검색량 추이, 상관·CCA(정준상관)
- **t-test & VIF**: 유의 피처 선별 + Stepwise VIF 다중공선성 제거 (전체 모델 VIF 전후 비교 포함)
- **모델 예측(비교1)·핵심 변수 선별(비교2)**: 공원별 표준 ML 비교 — Ridge · ElasticNet · GradientBoosting · RandomForest · ExtraTrees
- **신규 모델(HSKR)**: 직접 구현한 Hybrid Seasonal Kernel Ridge(계절 푸리에 + RBF 커널)를 공원별로 학습, 표준 ML과 동일 holdout 비교
- **해석**: SHAP(변수 기여), LIME(개별 예측 국소 해석)
- **불확실성·검증**: Conformal(예측 구간) · Bootstrap(성능 신뢰구간) · Nested CV(일반화 추정)

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run appnew.py
```

## 배포 (Streamlit Cloud)

- 진입점: **`appnew.py`**
- `model/*.pkl`이 numpy/sklearn 객체를 담으므로 `requirements.txt` 버전을 로컬과 동일하게 고정해야 Cloud에서 로드됩니다.

## 데이터·모델 파일

- `data/users.csv`: 서울시 한강공원 월별 이용객 현황 (2018-01 ~ 2024-02)
- `data/trend.xlsx`: 네이버 트렌드 검색량 (11개 공원 + 한강공원 통합)
- `model/fi_models.pkl`: 표준 ML 분석 번들 (하이퍼파라미터·VIF 피처)
- `model/fi_perpark.pkl`: 공원별 비교1·비교2 사전계산 번들 — `build_fi_perpark.py`로 생성
- `model/hskr_model.pkl`: 공원별 HSKR 결과 번들
- `hskr_model.py`: HSKR 모델 클래스 (pkl 언피클에 필요)
