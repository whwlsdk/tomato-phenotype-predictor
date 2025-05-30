# tomato-phenotype-predictor
# 🍅 Tomato Phenotype Predictor

AI 기반 유전형(GenoType) 데이터로 토마토의 주요 표현형(형질)을 예측하는 Streamlit 애플리케이션

##  기능 소개

- **3가지 모델 선택 가능**
  - MIR 기반 CatBoost (5000 SNP)
  - GWAS 기반 RandomForest (3000 SNP)
  - GWAS 기반 CatBoost (10000 SNP)

- **예측 대상 표현형**
  - 과실경도 (kg)
  - 과장 (mm)
  - 과중 (g)
  - 과폭 (mm)
  - 과피두께 (mm)
  - 당도 (%)

- **SHAP 기반 XAI 해석 기능**
  - 예측에 영향을 미친 상위 20개 SNP 시각화

## 📂 파일 업로드 가이드

업로드할 CSV 유전형 파일은 다음 기준을 따라야 합니다:

- 샘플 x SNP 형식
- 첫 번째 열 이름: `Genotype` 또는 `SampleID` 또는 `Unnamed: 0`
- SNP 컬럼 이름은 학습된 모델의 SNP 컬럼과 일치해야 함

## 실행 방법

### 1. 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
