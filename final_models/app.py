import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger

# =========================================================
# 1. CONFIGURATION
# =========================================================
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

st.set_page_config(
    page_title="Candida Albicans Virtual Screening",
    page_icon="🧬",
    layout="wide"
)

MODEL_DIR = "final_models"

AVAILABLE_MODELS = {
    "Stacking Ensemble": "stacking_ensemble_model.pkl",
    "Random Forest": "tuned_random_forest_model.pkl",
    "XGBoost": "tuned_xgboost_model.pkl",
    "Support Vector Machine (SVM)": "tuned_support_vector_machine_(svm)_model.pkl",
    "MLP Neural Network": "tuned_mlp_neural_network_model.pkl",
    "Logistic Regression": "tuned_logistic_regression_model.pkl",
    "KNN": "tuned_k-nearest_neighbors_(knn)_model.pkl",
    "AdaBoost": "tuned_adaboost_model.pkl"
}

CHUNK_SIZE = 10000


# =========================================================
# 2. LOAD PIPELINE
# =========================================================
@st.cache_resource
def load_pipeline_tools():
    tools = {}
    tools["scaler"] = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
    tools["var_thresh"] = joblib.load(os.path.join(MODEL_DIR, "variance_threshold_selector.pkl"))
    tools["feat_selector"] = joblib.load(os.path.join(MODEL_DIR, "model_feature_selector.pkl"))
    return tools


def load_model_file(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


pipeline = load_pipeline_tools()


# =========================================================
# 3. FINGERPRINTS
# =========================================================
def fingerprints_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
    maccs = MACCSkeys.GenMACCSKeys(mol)

    return np.concatenate([np.array(ecfp), np.array(fcfp), np.array(maccs)])


# =========================================================
# 4. SCREENING
# =========================================================
def run_screening(df, smiles_col, selected_models):

    all_results = []
    total_rows = len(df)
    num_chunks = (total_rows // CHUNK_SIZE) + 1
    prog = st.progress(0)

    for chunk_index, start in enumerate(range(0, total_rows, CHUNK_SIZE)):

        chunk = df.iloc[start:start + CHUNK_SIZE].copy()
        fps, keep_idx = [], []

        for i, s in enumerate(chunk[smiles_col]):
            fp = fingerprints_from_smiles(s)
            if fp is not None:
                fps.append(fp)
                keep_idx.append(chunk.index[i])

        if not fps:
            continue

        X_df = pd.DataFrame(fps, index=keep_idx)

        X_var = pipeline["var_thresh"].transform(X_df)
        X_feat = pipeline["feat_selector"].transform(X_var)
        X_final = pipeline["scaler"].transform(X_feat)

        results_batch = chunk.loc[keep_idx].copy()
        prob_cols = []

        for model_name in selected_models:
            model = load_model_file(AVAILABLE_MODELS[model_name])

            preds = model.predict(X_final)
            results_batch[f"{model_name}_Pred"] = preds

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_final)[:, 1]
                col = f"{model_name}_Prob"
                results_batch[col] = np.round(probs, 4)
                prob_cols.append(col)

        if prob_cols:
            results_batch["Consensus_Probability"] = results_batch[prob_cols].mean(axis=1)
            results_batch["Model_Vote"] = results_batch[[c for c in results_batch.columns if c.endswith("_Pred")]].sum(axis=1)

        all_results.append(results_batch)
        prog.progress((chunk_index + 1) / num_chunks)

    prog.empty()

    final_df = pd.concat(all_results)

    if "Consensus_Probability" in final_df.columns:
        final_df = final_df.sort_values("Consensus_Probability", ascending=False)

    return final_df.reset_index(drop=True)


# =========================================================
# 5. INTERFACE
# =========================================================

# ---- Styling ----
st.markdown("""
<style>
.app-title{
    font-size:32px;
    font-weight:700;
    color:#0F2B46;
}
.app-subtitle{
    font-size:18px;
    color:#2F4F4F;
    font-style:italic;
}
.hint-box{
    background-color:#F3F8FF;
    border-radius:8px;
    padding:10px 12px;
    border:1px solid #D6E4FF;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="app-title">Candida <i>albicans</i> Virtual Screening Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Machine learning–guided prediction of β-1,3-Glucan Synthase inhibitors</p>', unsafe_allow_html=True)

st.markdown("---")

# ---- Sidebar ----
st.sidebar.header("Data Source")

source = st.sidebar.radio(
    "Select Input Mode",
    ["Upload CSV (≤1GB Recommended)", "Load from Server Path (10GB+)"]
)

selected_models = st.sidebar.multiselect(
    "Select Models",
    list(AVAILABLE_MODELS.keys()),
    default=["Stacking Ensemble"]
)

df_raw = None


# ---- Upload mode ----
if source == "Upload CSV (≤1GB Recommended)":


    uploaded = st.sidebar.file_uploader("", type=["csv"])  # << NO LABEL

    if uploaded:
        df_raw = pd.read_csv(uploaded)


# ---- Server path mode ----
else:
    path = st.sidebar.text_input("Server File Path", "/data/library.csv")
    if os.path.exists(path):
        df_raw = pd.read_csv(path)
        st.sidebar.success("File Loaded Successfully")


# ---- Screening ----
if df_raw is not None and len(selected_models) > 0:

    smiles_col = next((c for c in df_raw.columns if "smile" in c.lower()), None)

    if smiles_col is None:
        st.error("No SMILES column found.")
    else:

        st.write("### Preview of Compound Library")
        st.dataframe(df_raw.head())

        if st.button("Start Virtual Screening", type="primary"):

            st.info("Processing — this may take time for large datasets…")

            results_df = run_screening(df_raw, smiles_col, selected_models)

            st.success("Screening Complete ✔")

            st.write("### Ranked Screening Output")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Ranked Results",
                csv,
                "ranked_screening_results.csv",
                "text/csv"
            )

else:
    st.info("Upload or select a dataset to begin.")
