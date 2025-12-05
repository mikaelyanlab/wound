import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shap
import matplotlib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import math

# -----------------------------------------------------
# CONFIG & DATA
# -----------------------------------------------------
st.set_page_config(page_title="Wound Ecology Explorer", layout="wide", page_icon="maggot")
st.title("Wound Ecology Explorer")
st.markdown("### Clinical decision support for chronic wounds & maggot debridement therapy")

uploaded = st.file_uploader("Upload wound dataset (.csv)", type="csv")
if uploaded is None:
    st.info("Upload your CSV to start.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {df.shape[0]:,} wounds")

# -----------------------------------------------------
# 1. SAFE ID EXCLUSION
# -----------------------------------------------------
id_patterns = [r"subject", r"patient", r"hadm", r"specimen", r"_id$", r"stay_id"]
id_cols = [c for c in df.columns if any(re.search(p, c, re.IGNORECASE) for p in id_patterns)]
df[id_cols] = df[id_cols].astype("category")

# -----------------------------------------------------
# 2. COLUMN DETECTION
# -----------------------------------------------------
microbe_cols = [
    c for c in df.columns
    if c.isupper() and pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() <= 3
]

elix_cols = [c for c in ["chf","valv","pulc","perivasc","htn","htnc","para","neuro","diab","diabwc",
                         "hypothy","renal","liver","ulcer","aids","lymph","mets","tumor","arth",
                         "coag","obes","wloss","fed","blane","dane","alcohol","drug","psycho","depre"]
             if c in df.columns]

wound_type_vars = [c for c in df.columns if c.startswith("wound_")]
ABX_NAME_COL = "abx_names_all" if "abx_names_all" in df.columns else None
ABX_INTERP_COL = "abx_interp_all" if "abx_interp_all" in df.columns else None
resistance_score_cols = [c for c in df.columns if "resistance_score" in c.lower()]

# -----------------------------------------------------
# 3. DERIVED MAGGOT CANDIDATE FLAG (proxy)
# -----------------------------------------------------
df["maggot_candidate"] = 0
if "days_open" in df.columns:
    df["maggot_candidate"] = df["maggot_candidate"] | (df["days_open"] > 60)
if any(c in df.columns for c in resistance_score_cols):
    rs = df[resistance_score_cols[0]].fillna(0)
    df["maggot_candidate"] = df["maggot_candidate"] | (rs >= 3)
if "PSEUDOMONAS_AERUGINOSA" in df.columns and "STAPHYLOCOCCUS_AUREUS" in df.columns:
    df["maggot_candidate"] = df["maggot_candidate"] | ((df["PSEUDOMONAS_AERUGINOSA"] + df["STAPHYLOCOCCUS_AUREUS"]) >= 1)

# -----------------------------------------------------
# SIDEBAR WITH PRESET QUESTIONS
# -----------------------------------------------------
st.sidebar.header("Quick Clinical Questions")
preset = st.sidebar.selectbox("Choose a question", [
    "Custom analysis",
    "Who fails to heal / close?",
    "Who has multidrug-resistant organisms?",
    "Who grows Pseudomonas aeruginosa?",
    "Who grows Staphylococcus aureus?",
    "Best candidates for maggot therapy"
])

# -----------------------------------------------------
# PRESET LOGIC → auto-select outcome + features
# -----------------------------------------------------
if preset != "Custom analysis":
    if "heal" in preset or "close" in preset:
        possible = [c for c in df.columns if any(k in c.lower() for k in ["heal","close","resolved","closure"])]
        target = possible[0] if possible else None
    elif "multidrug" in preset:
        target = next((c for c in df.columns if "mdr" in c.lower() or "multi_drug" in c.lower()), None)
    elif "Pseudomonas" in preset:
        target = "PSEUDOMONAS_AERUGINOSA"
    elif "Staphylococcus" in preset:
        target = "STAPHYLOCOCCUS_AUREUS"
    elif "maggot" in preset:
        target = "maggot_candidate"
    else:
        target = None

    use_microbes = "Who fails" not in preset
    use_comorb = True
    use_wound = "Who fails" in preset
    use_res = True
else:
    target = None
    use_microbes = use_comorb = use_wound = use_res = True

# -----------------------------------------------------
# PREDICTIVE MODELING (now with SHAP + presets)
# -----------------------------------------------------
st.header("Predictive Modeling")

# All true binary columns (including microbes!)
binary_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == 2 and c not in id_cols]

if target and target in binary_cols:
    chosen_target = target
else:
    chosen_target = st.selectbox("Or select any outcome", binary_cols, index=0)

c1 = st.checkbox("Microbes", value=use_microbes or st.session_state.get("c1", True), key="c1")
c2 = st.checkbox("Comorbidities", value=use_comorb or st.session_state.get("c2", True), key="c2")
c3 = st.checkbox("Wound type vars", value=use_wound or st.session_state.get("c3", False), key="c3")
c4 = st.checkbox("Resistance score", value=use_res or st.session_state.get("c4", True), key="c4")

if st.button("Run Model", type="primary"):
    # === run_model stays the same (included below) ===
    model, auc, importances, X_test, y_test = run_model(
        df, chosen_target, c1, c2, c3, c4
    )
    if model is None:
        st.stop()

    st.success(f"**AUC = {auc:.3f}**")

    # Top features bar chart
    fig, ax = plt.subplots(figsize=(10,6))
    importances.head(15).plot.bar(ax=ax)
    ax.set_title("Top 15 Predictive Features")
    st.pyplot(fig)

    # SHAP waterfall for a high-risk patient
    st.subheader("Why is this patient high-risk?")
    high_risk_idx = X_test[y_test == 1].index[0] if (y_test == 1).any() else X_test.index[0]
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(df.loc[high_risk_idx:high_risk_idx][importances.index])
    shap.waterfall(shap.Explanation(values=shap_val[1][0], base_values=explainer.expected_value[1],
                                    data=df.loc[high_risk_idx][importances.index]), show=True)
    st.pyplot(plt.gcf())
    plt.clf()

    # Download predictions
    preds = model.predict_proba(df[importances.index])[:, 1]
    df_out = df.copy()
    df_out["Predicted_Risk"] = preds
    csv = df_out.to_csv(index=False)
    st.download_button("Download full predictions", csv, "wound_predictions.csv", "text/csv")

# -----------------------------------------------------
# TARGET GENERATOR — now with MaggotScore
# -----------------------------------------------------
if ABX_NAME_COL and ABX_INTERP_COL and microbe_cols:
    st.header("Maggot Therapy Target Generator")
    top_k = st.slider("Antibiotics", 5, 40, 20)
    abx_names, heat_A, heat_B, heat_C = compute_microbe_abx_matrices(
        df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k
    )
    if abx_names:
        prev = df[microbe_cols].mean()
        A_score = heat_A.mean(axis=1)
        B_score = heat_B.clip(lower=0).mean(axis=1)
        C_score = heat_C.clip(lower=0).mean(axis=1)

        score = (0.5 * norm_series(A_score) +
                 0.3 * norm_series(B_score) +
                 0.2 * norm_series(C_score)).fillna(0)

        # Final MaggotScore = prevalence + resistance burden
        maggot_score = 0.4 * norm_series(prev) + 0.6 * score

        targets = pd.DataFrame({
            "Prevalence": prev,
            "Resistance Burden": A_score,
            "TargetScore": score,
            "MaggotScore (final)": maggot_score
        }).sort_values("MaggotScore (final)", ascending=False)

        st.dataframe(targets.head(20).style.bar(subset=["MaggotScore (final)"], color="#ff4b4b"))
        st.success("Top microbes = best candidates for Lucilia (maggot) therapy assays")

# -----------------------------------------------------
# HELPER FUNCTIONS (run_model + others)
# -----------------------------------------------------
def detect_leakage(X, y):
    leak = []
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0,1]
        if np.isfinite(corr) and abs(corr) > 0.95:
            leak.append(col)
    return leak

def run_model(df, target_col, m, c, w, r):
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int)
    if y.nunique() != 2:
        st.error("Not binary.")
        return None, None, None, None, None

    feats = []
    if m: feats += microbe_cols
    if c: feats += elix_cols
    if w: feats += wound_type_vars
    if r: feats += resistance_score_cols

    feats = [f for f in feats if f not in id_cols + [target_col]]
    X = df[feats].astype(float)

    leak = detect_leakage(X, y)
    if leak:
        st.warning(f"Leakage removed: {leak}")
        X = X.drop(columns=leak)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, auc, imp, X_test, y_test

# (compute_microbe_abx_matrices and norm_series unchanged — paste from previous version)

def compute_microbe_abx_matrices(df, microbe_cols, abx_name_col, abx_interp_col, top_k=20):
    # ← paste your existing working function here (same as before)
    # Omitted for brevity — just copy-paste from your last working script
    pass  # replace with full function

def norm_series(s):
    s = s.copy()
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

# DONE
st.caption("Wound Ecology Explorer — built for maggot therapy discovery")
