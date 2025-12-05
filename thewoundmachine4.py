# # The Wound Machine v4.0
# Aram Mikaelyan, Department of Entomology and Plant Pathology, NCSU | amikael@ncsu.edu | http://www.mikaelyanlab.com
# 
# Description:
#    Streamlit app for exploring the MIMIC wound dataset
#
#    The app is a scaling widget to visualize potential gigaton-scale impacts
#
# Reference:
#    Mikaelyan & Welsh, in prep.
###############
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import math

# -----------------------------------------------------
# PAGE & UPLOAD
# -----------------------------------------------------
st.set_page_config(page_title="Wound Ecology Explorer", layout="wide", page_icon="fly")
st.title("Wound Ecology Explorer")

uploaded = st.file_uploader("Upload your wound dataset (.csv)", type="csv")
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.success("Dataset loaded!")
st.write(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")

# -----------------------------------------------------
# 1. EXCLUDE ONLY TRUE ID COLUMNS
# -----------------------------------------------------
id_patterns = [r"subject", r"patient", r"hadm", r"specimen", r"_id$", r"stay_id"]
id_cols = [c for c in df.columns if any(re.search(p, c, re.IGNORECASE) for p in id_patterns)]

if id_cols:
    st.info(f"Excluded ID columns: {', '.join(id_cols)}")
    df[id_cols] = df[id_cols].astype("category")

# Safe quantitative numeric columns
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols]

# -----------------------------------------------------
# 2. COLUMN DETECTION
# -----------------------------------------------------
microbe_cols = [
    c for c in df.columns
    if c.isupper() and pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() <= 3
]

ELIX_COLS = ["chf","valv","pulc","perivasc","htn","htnc","para","neuro",
             "diab","diabwc","hypothy","renal","liver","ulcer","aids",
             "lymph","mets","tumor","arth","coag","obes","wloss","fed",
             "blane","dane","alcohol","drug","psycho","depre"]
elix_cols = [c for c in ELIX_COLS if c in df.columns]

wound_type_vars = [c for c in df.columns if c.startswith("wound_")]
ABX_NAME_COL = "abx_names_all" if "abx_names_all" in df.columns else None
ABX_INTERP_COL = "abx_interp_all" if "abx_interp_all" in df.columns else None
resistance_score_cols = [c for c in df.columns if "resistance_score" in c.lower()]

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def detect_leakage(X, y):
    leakage = []
    y_arr = y.values.astype(float)
    if y_arr.std() == 0: return leakage
    for col in X.columns:
        x = X[col].values.astype(float)
        if x.std() == 0: continue
        corr = np.corrcoef(x, y_arr)[0, 1]
        if np.isfinite(corr) and abs(corr) > 0.95:
            leakage.append(col)
    return leakage

def run_model(df, target_col, use_microbes=True, use_comorb=True,
              use_wound_types=False, use_resistance_score=True):
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int)
    if y.nunique() != 2:
        st.error("Outcome must be binary.")
        return

    feats = []
    if use_microbes: feats.extend(microbe_cols)
    if use_comorb: feats.extend(elix_cols)
    if use_wound_types: feats.extend(wound_type_vars)
    if use_resistance_score: feats.extend(resistance_score_cols)

    feats = [f for f in feats if f not in [target_col] + id_cols]
    forbidden = {"multi_drug_resistant": wound_type_vars + resistance_score_cols,
                 "infection_persistent_any_site": wound_type_vars}.get(target_col, [])
    feats = [f for f in feats if f not in forbidden]
    if not feats:
        st.error("No features selected.")
        return

    X = df[feats].astype(float)
    leak = detect_leakage(X, y)
    if leak:
        st.warning(f"Leakage removed: {leak}")
        X = X.drop(columns=leak)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=400, max_depth=10,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.success(f"AUC = {auc:.3f}")

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot.bar(ax=ax)
    ax.set_title("Top Features")
    st.pyplot(fig)

def compute_microbe_abx_matrices(df, microbe_cols, abx_name_col, abx_interp_col, top_k=20):
    # (same function as before – unchanged, works perfectly)
    abx_total_N = defaultdict(int); abx_total_R = defaultdict(int)
    N_present = defaultdict(int); R_present = defaultdict(int)
    for _, row in df.iterrows():
        names = row.get(abx_name_col); interps = row.get(abx_interp_col)
        if not (isinstance(names, str) and isinstance(interps, str)): continue
        names = [n.strip().upper() for n in names.split(",") if n.strip()]
        interps = [i.strip().upper() for i in interps.split(",") if i.strip()]
        if not names: continue
        L = min(len(names), len(interps))
        names, interps = names[:L], interps[:L]
        present = [m for m in microbe_cols if row[m] == 1]
        for n, i in zip(names, interps):
            abx_total_N[n] += 1
            res = 1 if i in ("R","RESISTANT","I","INTERMEDIATE") else 0
            if res: abx_total_R[n] += 1
            for m in present:
                N_present[(m,n)] += 1
                if res: R_present[(m,n)] += 1
    if not abx_total_N: return [], None, None, None
    top_abx = [a for a,_ in sorted(abx_total_N.items(), key=lambda x: -x[1])[:top_k]]
    A = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)
    B = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)
    C = pd.DataFrame(np.nan, index=microbe_cols, columns=top_abx)
    for m in microbe_cols:
        for a in top_abx:
            npres = N_present[(m,a)]; rpres = R_present[(m,a)]
            ntot = abx_total_N[a]; rtot = abx_total_R[a]
            if npres: A.loc[m,a] = rpres / npres
            nabs = ntot - npres; rabs = rtot - rpres
            OR = ((rpres+0.5)/(npres-rpres+0.5)) / ((rabs+0.5)/(nabs-rabs+0.5))
            B.loc[m,a] = math.log2(OR) if OR > 0 else np.nan
            p1 = rpres/npres if npres else 0
            p0 = rabs/nabs if nabs else 0
            if p1 and p0: C.loc[m,a] = math.log2(p1/p0)
    return top_abx, A, B, C

def norm_series(s):
    s = s.copy()
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

# -----------------------------------------------------
# SIDEBAR & MODULES
# -----------------------------------------------------
module = st.sidebar.radio("Module", [
    "Dataset Explorer", "Microbial Ecology", "Antibiotic Resistance Patterns",
    "Comorbidity Landscape", "Predictive Modeling", "Target Generator (Maggot Therapy)"
])

if module == "Dataset Explorer":
    st.header("Dataset Explorer")
    if numeric_cols:
        col = st.selectbox("Variable", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
        if len(numeric_cols) > 1:
            st.subheader("Correlation")
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
    else:
        st.info("No quantitative columns.")

elif module == "Microbial Ecology":
    st.header("Microbial Ecology")
    if not microbe_cols:
        st.warning("No uppercase binary microbe columns.")
    else:
        st.write(f"Detected {len(microbe_cols)} microbes")
        prev = df[microbe_cols].mean().sort_values(ascending=False)
        st.bar_chart(prev.head(30))
        if len(microbe_cols) >= 2:
            X = df[microbe_cols].fillna(0)
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            fig, ax = plt.subplots()
            ax.scatter(pcs[:,0], pcs[:,1], alpha=0.6, s=15)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            st.pyplot(fig)

elif module == "Antibiotic Resistance Patterns":
    st.header("Antibiotic Resistance Patterns")
    if not (ABX_NAME_COL and ABX_INTERP_COL and microbe_cols):
        st.warning("Missing required columns.")
    else:
        top_k = st.slider("Antibiotics", 5, 40, 20)
        method = st.radio("Method", ["A: % Resistant", "B: log2 OR", "C: log2 FC"], horizontal=True)
        names, A, B, C = compute_microbe_abx_matrices(df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k)
        if names:
            heat = A if "A" in method else B if "B" in method else C
            fig, ax = plt.subplots(figsize=(14,10))
            sns.heatmap(heat, cmap="viridis" if "A" in method else "coolwarm",
                        center=0 if "A" not in method else None, ax=ax)
            st.pyplot(fig)

elif module == "Comorbidity Landscape":
    st.header("Comorbidities")
    if elix_cols:
        prev = df[elix_cols].mean().sort_values(ascending=False)
        st.bar_chart(prev)

elif module == "Predictive Modeling":
    st.header("Predictive Modeling")
    binary_outcomes = [c for c in df.columns if df[c].nunique() == 2 and c not in id_cols + microbe_cols]
    if not binary_outcomes:
        st.warning("No binary outcomes found.")
    else:
        target = st.selectbox("Select outcome", binary_outcomes)
        c1 = st.checkbox("Microbes", True)
        c2 = st.checkbox("Comorbidities", True)
        c3 = st.checkbox("Wound types", False)
        c4 = st.checkbox("Resistance score", True)
        if st.button("Run Model"):
            run_model(df, target, c1, c2, c3, c4)

elif module == "Target Generator (Maggot Therapy)":
    st.header("Maggot Therapy Targets")
    if not (ABX_NAME_COL and ABX_INTERP_COL and microbe_cols):
        st.stop()
    top_k = st.slider("Antibiotics", 5, 40, 20)
    names, A, B, C = compute_microbe_abx_matrices(df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k)
    if not names: st.stop()
    score = (0.5 * norm_series(A.mean(axis=1)) +
             0.3 * norm_series(B.clip(lower=0).mean(axis=1)) +
             0.2 * norm_series(C.clip(lower=0).mean(axis=1))).fillna(0)
    res = pd.DataFrame({"Prevalence": df[microbe_cols].mean(), "TargetScore": score})
    res = res.sort_values("TargetScore", ascending=False)
    st.dataframe(res.head(25))
    fig, ax = plt.subplots()
    res["TargetScore"].head(20).plot.bar(ax=ax)
    st.pyplot(fig)
