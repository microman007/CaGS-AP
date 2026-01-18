# ============================================================
# CaGS-AP — RDKit-FREE Streamlit Application
# Stable for Streamlit Cloud Deployment
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CaGS-AP: β-1,3-glucan synthase Activity Predictor",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<style>
.header-box{
    text-align:center;
    padding-top:6px;
}
.app-title{
    font-size:30px;
    font-weight:700;
    line-height:1.25;
}
.app-sub{
    font-size:15px;
    color:grey;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='header-box'>", unsafe_allow_html=True)
    try:
        logo = Image.open("App_Logo.png")
        st.image(logo, width=1800)
    except:
        pass

    st.markdown("""
        <div class='app-title'>
            🧬 CaGS-AP: Candida albicans β-1,3-glucan synthase — Activity Predictor
        </div>
        <div class='app-sub'>
            Machine-learning virtual screening using precomputed molecular fingerprints
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# GLOBAL SETTINGS
# ============================================================
MODEL_DIR = "final_models"
CHUNK_SIZE = 10000

AVAILABLE_MODELS = {
    "Logistic Regression": "tuned_logistic_regression_model.pkl",
    "KNN": "tuned_k-nearest_neighbors_(knn)_model.pkl",
    "Support Vector Machine (SVM)": "tuned_support_vector_machine_(svm)_model.pkl",
    "MLP Neural Network": "tuned_mlp_neural_network_model.pkl"
}

# ============================================================
# SAFE LOADERS
# ============================================================
@st.cache_resource
def load_pipeline():
    tools = {}
    tools["scaler"] = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
    tools["var_thresh"] = joblib.load(os.path.join(MODEL_DIR, "variance_threshold_selector.pkl"))
    tools["feat_selector"] = joblib.load(os.path.join(MODEL_DIR, "model_feature_selector.pkl"))
    return tools


def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


def get_pipeline_safe():
    try:
        return load_pipeline()
    except Exception as e:
        st.error("❌ Failed to load preprocessing pipeline.")
        st.code(str(e))
        st.stop()

# ============================================================
# METRICS & CONSENSUS
# ============================================================
def compute_consensus(df):
    prob_cols = [c for c in df.columns if c.endswith("_Prob")]
    pred_cols = [c for c in df.columns if c.endswith("_Pred")]

    df["Mean_Probability"] = df[prob_cols].mean(axis=1)
    df["Probability_SD"] = df[prob_cols].std(axis=1)
    df["Model_Vote"] = df[pred_cols].sum(axis=1)

    return df


def confidence_label(sd):
    if sd < 0.05:
        return "High"
    elif sd < 0.15:
        return "Moderate"
    return "Low"

# ============================================================
# SCREENING ENGINE (CSV-BASED)
# ============================================================
def run_virtual_screening(df, feature_cols, selected_models):

    pipeline = get_pipeline_safe()
    results_all = []

    progress = st.progress(0.0)
    total = len(df)

    for start in range(0, total, CHUNK_SIZE):
        chunk = df.iloc[start:start + CHUNK_SIZE].copy()
        X = chunk[feature_cols]

        X = pipeline["var_thresh"].transform(X)
        X = pipeline["feat_selector"].transform(X)
        X = pipeline["scaler"].transform(X)

        res = chunk.copy()
        prob_columns = []

        for model_name in selected_models:
            model = load_model(AVAILABLE_MODELS[model_name])

            res[f"{model_name}_Pred"] = model.predict(X)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
                col = f"{model_name}_Prob"
                res[col] = np.round(prob, 4)
                prob_columns.append(col)

        if prob_columns:
            res["Consensus_Probability"] = res[prob_columns].mean(axis=1)

        results_all.append(res)
        progress.progress(min((start + len(chunk)) / total, 1.0))

    progress.empty()
    final_df = pd.concat(results_all).reset_index(drop=True)
    final_df = compute_consensus(final_df)
    final_df["Confidence"] = final_df["Probability_SD"].apply(confidence_label)

    return final_df.sort_values("Consensus_Probability", ascending=False)

# ============================================================
# SIDEBAR UI
# ============================================================
st.sidebar.header("Input Mode")

st.sidebar.info(
    "🔹 This cloud version uses **precomputed molecular fingerprints**.\n\n"
    "🔹 SMILES-to-descriptor conversion is available in the local/Docker version."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with fingerprints/descriptors",
    type=["csv"]
)

selected_models = st.sidebar.multiselect(
    "Select Models",
    list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())
)

# ============================================================
# MAIN WORKFLOW
# ============================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Auto-detect feature columns
    feature_cols = [c for c in df.columns if c.startswith("FP_")]

    if not feature_cols:
        st.error(
            "❌ No fingerprint columns detected.\n\n"
            "Expected columns like `FP_0, FP_1, FP_2, ...`"
        )
        st.stop()

    st.success(f"✅ Detected {len(feature_cols)} fingerprint features")

    if st.button("🚀 Start Virtual Screening"):
        with st.spinner("Running virtual screening..."):
            results = run_virtual_screening(df, feature_cols, selected_models)

        st.subheader("🏆 Screening Results")
        st.dataframe(results)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            file_name="CaGS_AP_screening_results.csv",
            mime="text/csv"
        )

else:
    st.info(
        "👈 Upload a CSV file containing **precomputed molecular fingerprints** "
        "to start virtual screening."
    )
