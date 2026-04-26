# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator  # NEW IMPORT
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CaGS-AP | CURAJ",
    page_icon="🧬",
    layout="wide"
)
# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>

/* ===== GLOBAL FONT OVERRIDE (CRITICAL) ===== */
/* Apply Georgia ONLY to app text */
.stApp,
.block-container,
.title,
.subtitle,
.hero,
.section-title,
p,
label,
h1, h2, h3, h4, h5, h6 {
    font-family: Georgia, serif !important;
}

/* ❌ EXCLUDE interactive widgets */
div[data-testid="stDataFrame"],
div[role="menu"],
div[role="menu"] *,
div[class*="Mui"],
div[class*="ag-"],
div[class*="slick"],
canvas {
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
    font-family: Georgia, serif !important;
}

/* Streamlit app background */
.stApp{
    background-image:url("background.png");
    background-size:cover;
    background-attachment:fixed;
}

/* Main content container */
.block-container{
    background:rgba(255,255,255,0.92);
    padding:1.8rem;
    border-radius:20px;
}

/* Titles */
.title{
    text-align:center;
    font-size:36px;
    font-weight:700;
    color:#1f3c88;
}

.subtitle{
    text-align:center;
    font-size:15px;
    color:#555;
}

/* Hero */
.hero{
    background:linear-gradient(90deg,#1f4e79,#3c8dbc);
    padding:22px;
    border-radius:16px;
    color:white;
    text-align:center;
    margin-bottom:15px;
}

/* Section titles */
.section-title{
    font-size:22px;
    font-weight:600;
    text-align:center;
    margin:10px 0 6px 0;
}

/* Workflow image */
.workflow-img img{
    max-height:280px;
    object-fit:contain;
}

/* Morphology images */
.morph-card img{
    max-height:110px;
    border-radius:12px;
    box-shadow:0 6px 16px rgba(0,0,0,0.2);
}

/* Info box spacing */
div[data-testid="stAlert"]{
    margin-top:8px !important;
    margin-bottom:14px !important;
    padding-bottom:12px !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("<div class='title'>CaGS-AP Info</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>FPGE Lab | Central University of Rajasthan</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h2>CaGS-AP</h2>
<h4>AI-Driven Biosolutions for Antifungal Drug Discovery</h4>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    ["ℹ️ CaGS-AP Overview", "🧪 Activity Predictor"],
    horizontal=True
)
# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_DIR = "final_models"
CHUNK_SIZE = 20000

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
# LOADERS (OPTIMIZED WITH CACHING)
# ===============================
@st.cache_resource
def load_pipeline():
    tools = {}
    tools["scaler"] = joblib.load(os.path.join(MODEL_DIR,"standard_scaler.pkl"))
    tools["var_thresh"] = joblib.load(os.path.join(MODEL_DIR,"variance_threshold_selector.pkl"))
    tools["feat_selector"] = joblib.load(os.path.join(MODEL_DIR,"model_feature_selector.pkl"))
    return tools

@st.cache_resource
def load_model_file(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

pipeline = load_pipeline()


# ===============================
# FINGERPRINTS (UPDATED for RDKit 2024+)
# ===============================
def fingerprints_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    
    # NEW: Use rdFingerprintGenerator instead of deprecated AllChem calls
    # 1. ECFP (Morgan, radius 2)
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    ecfp = mfgen.GetFingerprint(mol)

    # 2. FCFP (Morgan, radius 2, useFeatures=True)
    # Note: We use atomInvariantsGenerator to simulate feature-based invariants
    inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    mfgen_feat = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, atomInvariantsGenerator=inv_gen)
    fcfp = mfgen_feat.GetFingerprint(mol)

    # 3. MACCS Keys
    maccs = MACCSkeys.GenMACCSKeys(mol)
    
    return np.concatenate([np.array(ecfp), np.array(fcfp), np.array(maccs)])


# ===============================
# SCREENING ENGINE (OPTIMIZED)
# ===============================
def run_screening(df, smiles_col, models):

    all_results=[]
    total=len(df)
    prog=st.progress(0.0)

    # OPTIMIZATION: Load selected models once BEFORE the loop
    loaded_models = {m: load_model_file(AVAILABLE_MODELS[m]) for m in models}

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

        # Apply pipeline transformations
        X = pipeline["var_thresh"].transform(X)
        X = pipeline["feat_selector"].transform(X)
        X = pipeline["scaler"].transform(X)

        results = chunk.loc[keep].copy()
        pcols = []

        for m in models:
            # Use pre-loaded model from dictionary
            model = loaded_models[m]

            preds = model.predict(X)
            results[f"{m}_Pred"] = preds

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
                cname = f"{m}_Prob"
                results[cname] = np.round(prob, 4)
                pcols.append(cname)

        if pcols:
            results["Consensus_Probability"] = results[pcols].mean(axis=1)

        if i == 0:
            results.to_csv("temp_results.csv", index=False)
        else:
            results.to_csv("temp_results.csv", mode="a", header=False, index=False)

        prog.progress(min((i+len(chunk))/total, 1.0))

    prog.empty()

    if all_results:
        final = pd.read_csv("temp_results.csv")
        if "Consensus_Probability" in final:
            final = final.sort_values("Consensus_Probability", ascending=False)
        return final.reset_index(drop=True)
    else:
        return pd.DataFrame()


# ===============================
# METRICS & SCORING
# ===============================
def compute_consensus_metrics(results_df):
    pred_cols = [c for c in results_df.columns if c.endswith("_Pred")]
    prob_cols = [c for c in results_df.columns if c.endswith("_Prob")]

    if prob_cols:
        results_df["Mean_Prob"] = results_df[prob_cols].mean(axis=1)
        results_df["Prob_SD"] = results_df[prob_cols].std(axis=1)
    
    if pred_cols:
        results_df["Model_Vote"] = results_df[pred_cols].sum(axis=1)

    return results_df


def assign_confidence(row):
    if "Prob_SD" not in row:
        return "N/A"
    if row["Prob_SD"] < 0.05:
        return "High"
    elif row["Prob_SD"] < 0.15:
        return "Moderate"
    return "Low"


# ===============================
# SCAFFOLD SAR
# ===============================
def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except:
        return None
    return None


# ===============================
# PLOTTING
# ===============================
def plot_probability(results_df):
    if "Consensus_Probability" not in results_df:
        return
    
    probs = results_df["Consensus_Probability"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(probs, kde=True, bins=30, color="#1f77b4", edgecolor="black", ax=ax)
    ax.set_title("Distribution of Predicted Probabilities")
    plt.tight_layout()
    st.pyplot(fig)


def plot_heatmap(results_df):
    prob_cols = [c for c in results_df.columns if c.endswith("_Prob")]
    if not prob_cols:
        return

    data = results_df[prob_cols].head(30)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(data, cmap="viridis", ax=ax, annot=True, fmt=".2f")
    ax.set_title("Heatmap of Top 30 Predictions")
    plt.tight_layout()
    st.pyplot(fig)


# ===============================
# SINGLE-SMILES PREDICTOR
# ===============================
def single_smiles_predict(smiles, models):

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        st.error("❌ Invalid SMILES string.")
        return

    # ----- Show molecule -----
    st.image(Draw.MolToImage(mol, size=(300,300)), caption="Structure Preview")

    fp = fingerprints_from_smiles(smiles)
    if fp is None:
        st.error("Could not generate fingerprints.")
        return

    X = pd.DataFrame([fp])

    # Transform
    X = pipeline["var_thresh"].transform(X)
    X = pipeline["feat_selector"].transform(X)
    X = pipeline["scaler"].transform(X)

    prob_dict = {}
    votes = 0

    for m in models:
        model = load_model_file(AVAILABLE_MODELS[m])

        pred = int(model.predict(X)[0])
        votes += pred

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
            prob_dict[m] = prob
        else:
            prob_dict[m] = float(pred) # Fallback if no proba

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
    if scaffold:
        st.write(f"Scaffold: `{scaffold}`")

    st.write("### Model-wise Probabilities")
    st.table(pd.DataFrame(prob_dict, index=["Probability"]).T)


# ============================================================
# OVERVIEW PAGE
# ============================================================

def show_overview():

    st.markdown(
        "<div class='section-title'>Machine Learning Workflow</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='workflow-img'>", unsafe_allow_html=True)
    st.image("Graphical_Abstract_1200_dpi.png", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'><i>Candida albicans</i> Morphological Forms</div>",
        unsafe_allow_html=True
    )

    IMG_DIR = "images"

    cols = st.columns(3)

    forms = [
        ("Yeast_form.png", "Yeast Form"),
        ("Hyphal_form.png", "Hyphal Form"),
        ("Bioflim_form.png", "Biofilm")
    ]

    for col, (img, label) in zip(cols, forms):

        with col:

            st.image(
                os.path.join(IMG_DIR, img),
                width=230
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-weight:700;
                    font-size:18px;
                    font-family: Georgia, serif;
                    margin-top:-5px;
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# PREDICTOR PAGE
# ============================================================
def show_predictor():

    st.markdown("<div class='section-title'>CaGS-AP Activity Predictor</div>",unsafe_allow_html=True)

    st.sidebar.header("Input Mode")

    mode = st.sidebar.radio("Choose Option",["Upload CSV","Predict from SMILES"])

    models = st.sidebar.multiselect(
        "Select Models",
        list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys())
    )

    if mode == "Upload CSV":

        file = st.file_uploader("Upload CSV",type=["csv"])

        if file:

            # FIX 1: initial load
            df = pd.read_csv(file, low_memory=False)

            # FIX 2: handle Excel 'sep=,' issue
            if len(df.columns) > 0 and "sep=" in str(df.columns[0]).lower():
                file.seek(0)
                df = pd.read_csv(file, low_memory=False, skiprows=1)

            # FIX 3: clean column names
            df.columns = df.columns.str.strip()

            st.dataframe(df.head())

            # DEBUG (optional)
            st.write("Detected columns:", df.columns.tolist())

            # FIX 4: auto-detect SMILES column
            smiles_col = next((c for c in df.columns if "smile" in c.lower()), None)

            # FIX 5: fallback manual selection
            if smiles_col is None:
                st.warning("SMILES column not auto-detected. Please select manually.")
                smiles_col = st.selectbox("Select SMILES column", df.columns)

            # FIX 6: final validation
            if smiles_col not in df.columns:
                st.error("Invalid SMILES column selected.")
                st.stop()

            if st.button("Start Virtual Screening"):

                results = run_screening(df,smiles_col,models)

                results = compute_consensus_metrics(results)
                results["Confidence"] = results.apply(assign_confidence,axis=1)
                results["Scaffold"] = results[smiles_col].apply(get_scaffold)

                st.write("Rows processed:", len(results))
                st.dataframe(results.head(1000))

                plot_probability(results)
                plot_heatmap(results)

                st.download_button(
                    "Download Results",
                    results.to_csv(index=False),
                    "CaGS_AP_results.csv"
                )

    else:

        smiles = st.text_area("Paste SMILES here:")

        if st.button("Predict Activity"):
            single_smiles_predict(smiles,models)
# ============================================================
# ROUTER
# ============================================================

if page == "ℹ️ CaGS-AP Overview":
    show_overview()
else:
    show_predictor()

st.info(
    "**CaGS-AP** is an AI-driven platform for predicting inhibitors of "
    "*Candida albicans* **β-1,3-glucan synthase**, supporting antifungal drug discovery."
)


