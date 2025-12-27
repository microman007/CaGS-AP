# ===============================
# IMPORTS
# ===============================
import streamlit as st
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF


# ===============================
# CONFIG
# ===============================
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

st.set_page_config(
    page_title="CaGS-AP: Candida albicans β-1,3-glucan synthase Activity Predictor",
    page_icon="🧬",
    layout="wide"
)

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
        if isinstance(p, str):
            os.makedirs(p, exist_ok=True)

    return paths


# ===============================
# LOADERS
# ===============================
@st.cache_resource
def load_pipeline():
    tools = {}
    tools["scaler"] = joblib.load(os.path.join(MODEL_DIR,"standard_scaler.pkl"))
    tools["var_thresh"] = joblib.load(os.path.join(MODEL_DIR,"variance_threshold_selector.pkl"))
    tools["feat_selector"] = joblib.load(os.path.join(MODEL_DIR,"model_feature_selector.pkl"))
    return tools

def load_model_file(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

pipeline = load_pipeline()



# ===============================
# FINGERPRINTS
# ===============================
def fingerprints_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useFeatures=True)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return np.concatenate([np.array(ecfp),np.array(fcfp),np.array(maccs)])



# ===============================
# SCREENING ENGINE (Phase-1)
# ===============================
def run_screening(df, smiles_col, models):

    all_results=[]
    total=len(df)
    prog=st.progress(0.0)

    for i in range(0,total,CHUNK_SIZE):

        chunk=df.iloc[i:i+CHUNK_SIZE].copy()
        fps=[]
        keep=[]

        for j,s in enumerate(chunk[smiles_col]):
            fp=fingerprints_from_smiles(s)
            if fp is not None:
                fps.append(fp)
                keep.append(chunk.index[j])

        if not fps:
            continue

        X=pd.DataFrame(fps,index=keep)

        X=pipeline["var_thresh"].transform(X)
        X=pipeline["feat_selector"].transform(X)
        X=pipeline["scaler"].transform(X)

        results=chunk.loc[keep].copy()
        pcols=[]

        for m in models:
            model=load_model_file(AVAILABLE_MODELS[m])

            preds=model.predict(X)
            results[f"{m}_Pred"]=preds

            if hasattr(model,"predict_proba"):
                prob=model.predict_proba(X)[:,1]
                cname=f"{m}_Prob"
                results[cname]=np.round(prob,4)
                pcols.append(cname)

        if pcols:
            results["Consensus_Probability"]=results[pcols].mean(axis=1)

        all_results.append(results)

        prog.progress(min((i+len(chunk))/total,1.0))

    prog.empty()

    final=pd.concat(all_results)

    if "Consensus_Probability" in final:
        final=final.sort_values("Consensus_Probability",ascending=False)

    return final.reset_index(drop=True)



# ===============================
# Phase-6 Metrics
# ===============================
def compute_consensus_metrics(results_df):

    pred_cols = [c for c in results_df.columns if c.endswith("_Pred")]
    prob_cols = [c for c in results_df.columns if c.endswith("_Prob")]

    results_df["Mean_Prob"] = results_df[prob_cols].mean(axis=1)
    results_df["Prob_SD"] = results_df[prob_cols].std(axis=1)
    results_df["Model_Vote"] = results_df[pred_cols].sum(axis=1)

    return results_df


def assign_confidence(row):
    if row["Prob_SD"] < 0.05:
        return "High"
    elif row["Prob_SD"] < 0.15:
        return "Moderate"
    return "Low"



# ===============================
# Scaffold SAR
# ===============================
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return None



# ===============================
# Phase-2 Plot
# ===============================
def plot_probability(results_df, dirs):

    probs=results_df["Consensus_Probability"]

    fig,ax=plt.subplots(figsize=(8,5),dpi=300)
    sns.histplot(probs,kde=True,bins=30,color="#1f77b4",edgecolor="black",ax=ax)

    plt.tight_layout()
    st.pyplot(fig)



# ===============================
# Phase-3 Heatmap
# ===============================
def plot_heatmap(results_df, dirs):

    prob_cols=[c for c in results_df.columns if c.endswith("_Prob")]
    data=results_df[prob_cols].head(30)

    fig,ax=plt.subplots(figsize=(8,6),dpi=300)
    sns.heatmap(data,cmap="viridis",ax=ax)

    plt.tight_layout()
    st.pyplot(fig)



# ===============================
# Phase-4 Interpretation
# ===============================
def model_interpretation():
    st.info("""
• Higher probability = stronger predicted inhibitor  
• Consensus probability = model-averaged confidence
""")


# =========================================================
# 🚀 NEW — SINGLE-SMILES PREDICTOR (MATCHES CSV MODE)
# =========================================================
def single_smiles_predict(smiles, models):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string.")
        return

    # ----- Show molecule -----
    st.image(Draw.MolToImage(mol, size=(300,300)))

    fp = fingerprints_from_smiles(smiles)
    X = pd.DataFrame([fp])

    X = pipeline["var_thresh"].transform(X)
    X = pipeline["feat_selector"].transform(X)
    X = pipeline["scaler"].transform(X)

    prob_dict = {}
    votes = 0

    for m in models:
        model = load_model_file(AVAILABLE_MODELS[m])

        pred = int(model.predict(X)[0])
        votes += pred

        prob = float(model.predict_proba(X)[0,1])
        prob_dict[m] = prob

    probs = list(prob_dict.values())
    mean_prob = np.mean(probs)
    sd_prob = np.std(probs)

    # ---- Confidence ----
    if sd_prob < 0.05:
        conf = "High"
    elif sd_prob < 0.15:
        conf = "Moderate"
    else:
        conf = "Low"

    scaffold = get_scaffold(smiles)

    # ---- Display ----
    st.success(f"Predicted Activity Probability: **{mean_prob:.4f}**")
    st.write(f"Model Vote: **{votes}/{len(models)}**")
    st.write(f"Std Dev: **{sd_prob:.4f}**")
    st.write(f"Confidence: **{conf}**")
    st.write(f"Scaffold: `{scaffold}`")

    st.write("### Model-wise Probabilities")
    st.table(pd.DataFrame(prob_dict,index=["Probability"]).T)



# ===============================
# UI
# ===============================
st.title("🧬 CaGS-AP: Candida albicans β-1,3-glucan synthase — Activity Predictor")
st.caption("Machine-learning prediction of β-1,3-Glucan Synthase inhibitors")
st.markdown("---")

dirs=get_output_dirs()


# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Input Mode")

mode=st.sidebar.radio("Choose Option",["Upload CSV","Predict from SMILES"])

models = st.sidebar.multiselect(
    "Select Models",
    list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())
)


# ===============================
# MODE-1
# ===============================
if mode=="Upload CSV":

    up=st.sidebar.file_uploader("Upload CSV",type=["csv"])

    if up:
        df=pd.read_csv(up)

        smiles_col = next((c for c in df.columns if "smile" in c.lower()), None)

        if st.button("Start Virtual Screening"):

            results = run_screening(df, smiles_col, models)
            results = compute_consensus_metrics(results)
            results["Confidence"] = results.apply(assign_confidence, axis=1)
            results["Scaffold"] = results[smiles_col].apply(get_scaffold)

            st.dataframe(results)


# ===============================
# MODE-2
# ===============================
else:
    st.subheader("🔍 Predict Activity from SMILES")
    smiles_input = st.text_area("Paste SMILES here:")

    if st.button("Predict Activity"):
        single_smiles_predict(smiles_input, models)

