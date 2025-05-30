import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt

# 표현형 및 모델 파일 경로 정의
TRAIT_MODELS = {
    "과실경도 (kg)": {
        "MIR (CatBoost_5000개)": "./과실경도 (kg)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./과실경도_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./과실경도_catboost(gwas)_model.cbm"
    },
    "과장 (mm)": {
        "MIR (CatBoost_5000개)": "./과장 (mm)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./과장_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./과장_catboost(gwas)_model.cbm"
    },
    "과중 (g)": {
        "MIR (CatBoost_5000개)": "./과중 (g)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./과중_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./과중_catboost(gwas)_model.cbm"
    },
    "과폭 (mm)": {
        "MIR (CatBoost_5000개)": "./과폭 (mm)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./과폭_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./과폭_catboost(gwas)_model.cbm"
    },
    "과피두께 (mm)": {
        "MIR (CatBoost_5000개)": "./과피두께 (mm)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./과피두께_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./과피두께_catboost(gwas)_model.cbm"
    },
    "당도 (%)": {
        "MIR (CatBoost_5000개)": "./당도 (%)_catboost_model.cbm",
        "GWAS (RandomForest_3000개)": "./당도_rf_model.pkl",
        "GWAS (CatBoost_10000개)": "./당도_catboost(gwas)_model.cbm"
    }
}

st.title("토마토 유전형 기반 표현형 예측기")

# 📘 사용법 안내
with st.expander("ℹ️ 사용법 안내 보기"):
    st.markdown("""
    **1. 모델 선택**
    - `MIR (CatBoost_5000개)`: 상호 정보량 기반 중요 SNP 5000개를 이용한 CatBoost 모델
    - `GWAS (RandomForest_3000개)`: GWAS 기반 중요 SNP 3000개를 이용한 RandomForest 모델
    - `GWAS (CatBoost_10000개)`: GWAS 기반 중요 SNP 10000개를 이용한 CatBoost 모델

    **2. 파일 업로드**
    - 전처리된 유전형 CSV 파일을 업로드해야 합니다 (샘플 x SNP 형태, SNP 컬럼명 일치 필요)

    **3. 예측 결과 확인**
    - 모든 표현형에 대해 예측값이 표시되며, CSV로 다운로드도 가능합니다

    **🔺주의사항**
    - 업로드한 유전형 데이터에 **모델 학습에 사용된 SNP가 많이 누락되었을 경우**, 예측 정확도가 **크게 감소할 수 있습니다**.
    - 가능한 한 모델에 맞는 SNP 구성을 포함한 유전형 파일을 사용해주세요.
    """)

model_options = [
    "MIR (CatBoost_5000개)",
    "GWAS (RandomForest_3000개)",
    "GWAS (CatBoost_10000개)"
]
selected_model_type = st.radio("⚙️ 분석 기준 및 모델 선택", model_options)

st.markdown("""      
💡 **분석 기준 안내**
- `GWAS` : 통계적 유의성(p-value) 중심 → **형질 원인 유전자 탐색**에 유리
- `MIR` : 비선형 정보량 중심 → **정확한 예측 및 실용적 선발**에 유리

""")

uploaded_file = st.file_uploader("📂 유전형 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    uploaded_file.seek(0)
    geno_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")

    for index_col in ["Genotype", "SampleID", "Unnamed: 0"]:
        if index_col in geno_raw.columns:
            geno_raw = geno_raw.set_index(index_col)
            break
    else:
        st.error("샘플 ID 열은 'Genotype', 'SampleID', 또는 'Unnamed: 0' 중 하나여야 합니다.")
        st.stop()

    tab1, tab2 = st.tabs(["📈 예측 결과", "🧬 SHAP 해석"])
    all_results = {}
    shap_outputs = {}

    with tab1:
        for trait, paths in TRAIT_MODELS.items():
            model_path = paths[selected_model_type]
            if not os.path.exists(model_path):
                st.warning(f"❌ 모델 없음: {trait} - {selected_model_type}")
                continue

            if "CatBoost" in selected_model_type:
                model = CatBoostRegressor()
                model.load_model(model_path)
                model_features = model.feature_names_
            else:
                model = joblib.load(model_path)
                model_features = model.feature_names_in_

            X = geno_raw.reindex(columns=model_features, fill_value=0)
            missing_snps = [snp for snp in model_features if snp not in geno_raw.columns]
            st.warning(f"📉 {trait} - 누락 SNP 수: {len(missing_snps)} / {len(model_features)}")

            preds = model.predict(X)
            all_results[trait] = preds
            shap_outputs[trait] = (model, X)

        result_df = pd.DataFrame(all_results, index=geno_raw.index)
        st.subheader("🎯 전체 예측 결과")
        st.dataframe(result_df)

        csv = result_df.to_csv().encode("utf-8-sig")
        st.download_button("⬇️ 예측 결과 다운로드", data=csv, file_name="예측결과.csv")

    with tab2:
        selected_trait = st.selectbox("🔬 SHAP 해석할 표현형 선택", list(shap_outputs.keys()))
        model, X = shap_outputs[selected_trait]

        if "CatBoost" in selected_model_type:
            shap_values = model.get_feature_importance(Pool(X), type="ShapValues")
            shap_values = shap_values[:, :-1]
            mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            mean_shap = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(mean_shap)[::-1][:20]
        top_snp = np.array(X.columns)[top_idx]
        top_val = mean_shap[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_snp[::-1], top_val[::-1])
        ax.set_title(f"{selected_trait} - SHAP 영향력 상위 20개 SNP")
        st.pyplot(fig)
