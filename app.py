# ===============================
# IMPORTS
# ===============================
import streamlit as st
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import sys
import numpy as np

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.umath"] = np.core.umath

import pandas as pd
import joblib

import os
import datetime
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF


# ===============================
# HEADER — LOGO + FULL-WIDTH TITLE
# ===============================
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
        st.image(logo, width=2000)
    except:
        pass

    st.markdown("""
        <div class='app-title'>
            🧬 CaGS-AP: Candida albicans β-1,3-glucan synthase — Activity Predictor
        </div>
        <div class='app-sub'>
            Machine-learning prediction of β-1,3-glucan synthase inhibitors
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

MODEL_DIR = "final_models"
CHUNK_SIZE = 10000

AVAILABLE_MODELS = {
    "Logistic Regression": "tuned_logistic_regression_model.pkl",
    "KNN": "tuned_k-nearest_neighbors_(knn)_model.pkl",
    "Support Vector Machine (SVM)": "tuned_support_vector_machine_(svm)_model.pkl",
    "MLP Neural Network": "tuned_mlp_neural_network_model.pkl"
}


# ===============================
# OUTPUT DIRECTORIES
# ===============================
def get_output_dirs():
    base = "Documents"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    paths = {
        "base": base,
        "ranked": os.path.join(base, "Ranked_Results"),
        "plots": os.path.join(base, "Plots"),
        "reports": os.path.join(base, "Reports"),
        "timestamp": ts
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths


# ===============================
# LOADERS
# ===============================
@st.cache_resource
def load_pipeline():
    tools = {}
    tools["scaler"] = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
    tools["var_thresh"] = joblib.load(os.path.join(MODEL_DIR, "variance_threshold_selector.pkl"))
    tools["feat_selector"] = joblib.load(os.path.join(MODEL_DIR, "model_feature_selector.pkl"))
    return tools


def load_model_file(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

# ===============================
# NumPy backward-compatibility patch
# ===============================
import sys
import numpy as np

# Alias old NumPy internal paths used by legacy pickles
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.umath"] = np.core.umath

pipeline = load_pipeline()


# ===============================
# FINGERPRINTS
# ===============================
def fingerprints_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return np.concatenate([np.array(ecfp), np.array(fcfp), np.array(maccs)])


# ===============================
# SCREENING ENGINE
# ===============================
def run_screening(df, smiles_col, models):

    all_results = []
    total = len(df)
    prog = st.progress(0.0)

    for i in range(0, total, CHUNK_SIZE):

        chunk = df.iloc[i:i+CHUNK_SIZE].copy()
        fps = []
        keep = []

        for j, s in enumerate(chunk[smiles_col]):
            fp = fingerprints_from_smiles(s)
            if fp is not None:
                fps.append(fp)
                keep.append(chunk.index[j])

        if not fps:
            continue

        X = pd.DataFrame(fps, index=keep)

        X = pipeline["var_thresh"].transform(X)
        X = pipeline["feat_selector"].transform(X)
        X = pipeline["scaler"].transform(X)

        results = chunk.loc[keep].copy()
        pcols = []

        for m in models:
            model = load_model_file(AVAILABLE_MODELS[m])
            results[f"{m}_Pred"] = model.predict(X)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
                cname = f"{m}_Prob"
                results[cname] = np.round(prob, 4)
                pcols.append(cname)

        if pcols:
            results["Consensus_Probability"] = results[pcols].mean(axis=1)

        all_results.append(results)
        prog.progress(min((i + len(chunk)) / total, 1.0))

    prog.empty()

    final = pd.concat(all_results)
    if "Consensus_Probability" in final:
        final = final.sort_values("Consensus_Probability", ascending=False)

    return final.reset_index(drop=True)


# ===============================
# METRICS
# ===============================
def compute_consensus_metrics(df):
    prob_cols = [c for c in df.columns if c.endswith("_Prob")]
    pred_cols = [c for c in df.columns if c.endswith("_Pred")]
    df["Mean_Prob"] = df[prob_cols].mean(axis=1)
    df["Prob_SD"] = df[prob_cols].std(axis=1)
    df["Model_Vote"] = df[pred_cols].sum(axis=1)
    return df


def assign_confidence(row):
    if row["Prob_SD"] < 0.05:
        return "High"
    elif row["Prob_SD"] < 0.15:
        return "Moderate"
    return "Low"


# ===============================
# SCAFFOLD
# ===============================
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return None


# ===============================
# SINGLE SMILES
# ===============================
def single_smiles_predict(smiles, models):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string.")
        return

    st.success("Molecule parsed successfully")
    st.info("⚠️ Molecular structure visualization is disabled on Streamlit Cloud.")

    fp = fingerprints_from_smiles(smiles)
    X = pd.DataFrame([fp])

    X = pipeline["var_thresh"].transform(X)
    X = pipeline["feat_selector"].transform(X)
    X = pipeline["scaler"].transform(X)

    probs = {}
    votes = 0

    for m in models:
        model = load_model_file(AVAILABLE_MODELS[m])
        pred = int(model.predict(X)[0])
        votes += pred
        probs[m] = float(model.predict_proba(X)[0, 1])

    mean_prob = np.mean(list(probs.values()))
    sd_prob = np.std(list(probs.values()))

    conf = "High" if sd_prob < 0.05 else "Moderate" if sd_prob < 0.15 else "Low"

    st.success(f"Predicted Activity Probability: **{mean_prob:.4f}**")
    st.write(f"Model Vote: **{votes}/{len(models)}**")
    st.write(f"Std Dev: **{sd_prob:.4f}**")
    st.write(f"Confidence: **{conf}**")
    st.write(f"Scaffold: `{get_scaffold(smiles)}`")
    st.table(pd.DataFrame(probs, index=["Probability"]).T)


# ===============================
# UI
# ===============================
dirs = get_output_dirs()

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose Option", ["Upload CSV", "Predict from SMILES"])

models = st.sidebar.multiselect(
    "Select Models",
    list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())
)

if mode == "Upload CSV":

    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if up:
        df = pd.read_csv(up)
        smiles_col = next((c for c in df.columns if "smile" in c.lower()), None)

        if smiles_col is None:
            st.error("No SMILES column found.")
        elif st.button("Start Virtual Screening"):
            res = run_screening(df, smiles_col, models)
            res = compute_consensus_metrics(res)
            res["Confidence"] = res.apply(assign_confidence, axis=1)
            res["Scaffold"] = res[smiles_col].apply(get_scaffold)
            st.dataframe(res)

elif mode == "Predict from SMILES":

    st.subheader("🔍 Predict Activity from SMILES")
    smiles_input = st.text_area("Paste SMILES here:")

    if st.button("Predict Activity"):
        if smiles_input.strip():
            st.code(f"SMILES: {smiles_input}")
            single_smiles_predict(smiles_input, models)
        else:
            st.warning("Please enter a SMILES string.")

