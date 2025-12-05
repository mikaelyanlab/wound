import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import math

# -----------------------------------------------------
# APP CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Wound Ecology Explorer", layout="wide", page_icon="maggot")
st.title("Wound Ecology Explorer")
st.markdown("**Clinical decision support for chronic wounds & maggot debridement therapy**")

# -----------------------------------------------------
# DATA UPLOAD
# -----------------------------------------------------
uploaded = st.file_uploader("Upload your wound dataset (.csv)", type="csv")
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {df.shape[0]:,} wounds × {df.shape[1]} columns")
st.dataframe(df.head(3))

# -----------------------------------------------------
# 1. EXCLUDE TRUE ID COLUMNS
# -----------------------------------------------------
id_patterns = [r"subject", r"patient", r"hadm", r"specimen", r"_id$", r"stay_id"]
id_cols = [c for c in df.columns if any(re.search(p, c, re.IGNORECASE) for p in id_patterns)]
if id_cols:
    st.info(f"Excluded ID columns: {', '.join(id_cols[:10])}{'...' if len(id_cols)>10 else ''}")
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
# 3. DERIVED: MAGGOT CANDIDATE PROXY
# -----------------------------------------------------
df["maggot_candidate"] = 0
if "days_open" in df.columns:
    df["maggot_candidate"] = df["maggot_candidate"] | (df["days_open"] > 60)
if resistance_score_cols:
    rs = df[resistance_score_cols[0]].fillna(0)
    df["maggot_candidate"] = df["maggot_candidate"] | (rs >= 3)
if all(c in df.columns for c in ["PSEUDOMONAS_AERUGINOSA", "STAPHYLOCOCCUS_AUREUS"]):
    df["maggot_candidate"] = df["maggot_candidate"] | (df[["PSEUDOMONAS_AERUGINOSA", "STAPHYLOCOCCUS_AUREUS"]].sum(axis=1) >= 1)

# -----------------------------------------------------
# SIDEBAR: QUICK CLINICAL QUESTIONS
# -----------------------------------------------------
st.sidebar.header("Quick Clinical Questions")
preset = st.sidebar.selectbox("Select a question", [
    "Custom analysis",
    "Who fails to heal?",
    "Who has MDR organisms?",
    "Who grows Pseudomonas aeruginosa?",
    "Who grows Staphylococcus aureus?",
    "Best candidates for maggot therapy"
], index=0)

if preset != "Custom analysis":
    if "heal" in preset:
        heal_cols = [c for c in df.columns if any(k in c.lower() for k in ["heal","close","resolved","closure"])]
        target = heal_cols[0] if heal_cols else None
    elif "MDR" in preset:
        target = next((c for c in df.columns if "mdr" in c.lower() or "multi_drug" in c.lower()), None)
    elif "Pseudomonas" in preset:
        target = "PSEUDOMONAS_AERUGINOSA"
    elif "Staphylococcus" in preset:
        target = "STAPHYLOCOCCUS_AUREUS"
    elif "maggot" in preset:
        target = "maggot_candidate"
    else:
        target = None

    use_microbes = "heal" not in preset
    use_comorb = True
    use_wound = "heal" in preset
    use_res = True
else:
    target = None
    use_microbes = use_comorb = use_wound = use_res = True

# -----------------------------------------------------
# PREDICTIVE MODELING
# -----------------------------------------------------
st.header("Predictive Modeling")

binary_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == 2 and c not in id_cols]

chosen_target = st.selectbox("Select outcome", binary_cols, index=binary_cols.index(target) if target and target in binary_cols else 0)

col1, col2 = st.columns(2)
with col1:
    c1 = st.checkbox("Microbes", value=use_microbes, key="c1")
    c2 = st.checkbox("Comorbidities", value=use_comorb, key="c2")
with col2:
    c3 = st.checkbox("Wound type vars", value=use_wound, key="c3")
    c4 = st.checkbox("Resistance score", value=use_res, key="c4")

if st.button("Run Model", type="primary"):
    model, auc, imp, X_test, y_test = run_model(df, chosen_target, c1, c2, c3, c4)
    if model is None:
        st.stop()

    st.success(f"**AUC = {auc:.3f}**")

    st.subheader("Top Predictive Features")
    fig, ax = plt.subplots(figsize=(10,6))
    imp.head(15).plot.bar(ax=ax, color="teal")
    st.pyplot(fig)

    st.subheader("Why is this patient high-risk?")
    high_risk_idx = X_test[y_test == 1].index[0] if (y_test == 1).any() else X_test.index[0]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(df.loc[high_risk_idx:high_risk_idx, imp.index])
    shap.waterfall(shap.Explanation(
        values=shap_vals[1][0],
        base_values=explainer.expected_value[1],
        data=df.loc[high_risk_idx, imp.index]
    ), show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    preds = model.predict_proba(df[imp.index])[:, 1]
    df_out = df.copy()
    df_out["Predicted_Risk"] = preds
    st.download_button("Download full predictions", df_out.to_csv(index=False), "wound_predictions.csv", "text/csv")

# -----------------------------------------------------
# MAGGOT THERAPY TARGET GENERATOR
# -----------------------------------------------------
if ABX_NAME_COL and ABX_INTERP_COL and microbe_cols:
    st.header("Maggot Therapy Target Generator")
    top_k = st.slider("Antibiotics to analyze", 5, 40, 20)
    abx_names, heat_A, heat_B, heat_C = compute_microbe_abx_matrices(df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k)
    if abx_names:
        prev = df[microbe_cols].mean()
        A_score = heat_A.mean(axis=1)
        B_score = heat_B.clip(lower=0).mean(axis=1)
        C_score = heat_C.clip(lower=0).mean(axis=1)

        resistance_score = (0.5 * norm_series(A_score) +
                            0.3 * norm_series(B_score) +
                            0.2 * norm_series(C_score)).fillna(0)

        maggot_score = 0.4 * norm_series(prev) + 0.6 * resistance_score

        targets = pd.DataFrame({
            "Prevalence": prev.round(3),
            "Resistance Burden": A_score.round(3),
            "MaggotScore": maggot_score.round(3)
        }).sort_values("MaggotScore", ascending=False)

        st.dataframe(targets.head(20).style.bar(subset=["MaggotScore"], color="#ff4b4b"))
        st.success("**Top microbes = best candidates for maggot therapy**")

# -----------------------------------------------------
# FULLY IMPLEMENTED FUNCTIONS
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
        st.error("Outcome must be binary (0/1).")
        return None, None, None, None, None

    feats = []
    if m: feats += microbe_cols
    if c: feats += elix_cols
    if w: feats += wound_type_vars
    if r: feats += resistance_score_cols

    feats = [f for f in feats if f not in id_cols + [target_col]]
    if not feats:
        st.error("No features selected.")
        return None, None, None, None, None

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

def compute_microbe_abx_matrices(df, microbe_cols, abx_name_col, abx_interp_col, top_k=20):
    abx_total_N = defaultdict(int)
    abx_total_R = defaultdict(int)
    N_present = defaultdict(int)
    R_present = defaultdict(int)

    for _, row in df.iterrows():
        names = row.get(abx_name_col)
        interps = row.get(abx_interp_col)
        if not (isinstance(names, str) and isinstance(interps, str)):
            continue
        names = [n.strip().upper() for n in names.split(",") if n.strip()]
        interps = [i.strip().upper() for i in interps.split(",") if i.strip()]
        if not names:
            continue
        L = min(len(names), len(interps))
        names, interps = names[:L], interps[:L]
        present = [m for m in microbe_cols if row[m] == 1]
        for abx, interp in zip(names, interps):
            abx_total_N[abx] += 1
            is_r = interp in ("R", "RESISTANT", "I", "INTERMEDIATE")
            if is_r:
                abx_total_R[abx] += 1
            for m in present:
                N_present[(m, abx)] += 1
                if is_r:
                    R_present[(m, abx)] += 1

    if not abx_total_N:
        return [], None, None, None

    top_abx = [a for a, _ in sorted(abx_total_N.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    heat_A = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)
    heat_B = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)
    heat_C = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)

    for m in microbe_cols:
        for a in top_abx:
            npres = N_present.get((m, a), 0)
            rpres = R_present.get((m, a), 0)
            ntot = abx_total_N[a]
            rtot = abx_total_R[a]
            nabs = ntot - npres
            rabs = rtot - rpres

            if npres > 0:
                heat_A.loc[m, a] = rpres / npres

            OR = ((rpres + 0.5)/(npres - rpres + 0.5)) / ((rabs + 0.5)/(nabs - rabs + 0.5))
            heat_B.loc[m, a] = math.log2(OR) if OR > 0 else np.nan

            p1 = rpres / npres if npres > 0 else 1e-6
            p0 = rabs / nabs if nabs > 0 else 1e-6
            heat_C.loc[m, a] = math.log2(p1 / p0)

    return top_abx, heat_A, heat_B, heat_C

def norm_series(s):
    s = s.copy()
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

st.caption("Wound Ecology Explorer — built for maggot therapy discovery")
