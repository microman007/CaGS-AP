# CaGS-AP : Candida albicans β-1,3-Glucan Synthase — Activity Predictor

CaGS-AP is a machine-learning powered platform for predicting the inhibitory activity of small molecules against **Candida albicans β-1,3-glucan synthase (CaGS)** — a clinically validated antifungal drug target.

This tool enables **virtual screening, hit-prioritization, and activity confidence assessment** using an ensemble of optimized machine-learning classifiers.

---

## 🏛 Affiliation

**Fungal Biology Laboratory**  
**Central University of Rajasthan, India**

**Authors**
- *Arvind V. Kayande*
- *Prof. Gajanan B. Zore*

---

## 📌 Application Overview

CaGS-AP allows users to:

✔ Upload chemical datasets (SMILES format)  
✔ Predict CaGS inhibitory activity  
✔ Compute model-consensus probability  
✔ Rank hits automatically  
✔ Assess prediction confidence  
✔ Perform scaffold-level SAR analysis  
✔ Save publication-ready plots & reports  

The tool is implemented in **Python + Streamlit**, supporting both **batch screening** and **single-molecule prediction** modes.
---

## 📥 Download & Run

### 1️⃣ Clone the repository
git clone https://github.com/microman007/CaGS-AP.git
cd CaGS-AP
### 2️⃣ Install dependencies
pip install -r requirements.txt
### 3️⃣ Run the App
streamlit run app.py
The app will open in your browser automatically.
---
## 📂 Input Format

Upload a `.csv` file containing a column with **SMILES** strings.

Example: SMILES
CCOc1ccc2nc(SCc3ccccc3)sc2c1CCC(=O)NCCC1=CNc2ccccc21

The app will output predicted activity & ranked probability.

---

## 📊 Key Features

- Ensemble ML prediction  
- Consensus probability scoring  
- Hit ranking  
- Probability distribution visualization  
- Model vote analysis  
- Scaffold-level SAR  
- High-resolution figure export  
- Auto-generated reports  

---

## ⚖ License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute with citation.

---

## 🧪 Intended Use

This tool is developed **for academic research & drug discovery workflow support**.  
It does not replace experimental validation.

---

## 🙏 Acknowledgement

We thank **Central University of Rajasthan** for infrastructure support.

---

## 📧 Contact

For queries, please contact:

📩 *avindkayande0007@gmail.com*  
📩 *2022phdbt010@curaj.ac.in*

---

⭐ If you find CaGS-AP useful, please consider citing our work (citation coming soon).


