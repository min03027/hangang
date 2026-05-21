# 🏞️ 한강공원 이용객 분석 대시보드

서울시 한강공원 11개소의 월별 이용객 현황과 네이버 검색 트렌드를 분석하는 Streamlit 대시보드입니다.

## 기능

- **공원별 EDA**: 계절별·월별 이용 패턴, 검색량 추이
- **t-test & VIF**: 유의 피처 선별 + Stepwise VIF 다중공선성 제거
- **ML 모델 비교**: Ridge, ElasticNet, GradientBoosting, XGBoost, Stacking
- **진단 도표**: 잔차 도표 4종, Q-Q Plot, Learning Curve
- **Conformal Prediction**: 90% 예측 구간 시각화
- **SHAP**: 변수 기여도 해석
- **LSTM**: 시계열 딥러닝 예측 (참고용)

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포 (Streamlit Cloud)

1. 이 레포를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. 레포 연결 → 자동 배포

## 데이터

- `data/이용객.csv`: 서울시 한강공원 월별 이용객 현황 (2018~2024)
- `data/트렌드.xlsx`: 네이버 트렌드 검색량 (11개 공원 + 한강공원 통합)
