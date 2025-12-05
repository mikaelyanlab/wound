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
# PAGE CONFIG & UPLOAD
# -----------------------------------------------------
st.set_page_config(page_title="Wound Ecology Explorer", layout="wide", page_icon="fly")
st.title("Wound Ecology Explorer")
st.write("Upload a wound dataset CSV to explore microbiology, resistance, and maggot therapy targets.")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded is None:
    st.info("Awaiting upload...")
    st.stop()

df = pd.read_csv(uploaded)
st.success("Dataset loaded!")
st.write(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")
st.dataframe(df.head())

# -----------------------------------------------------
# 1. EXCLUDE TRUE ID COLUMNS ONLY (fixed & safe)
# -----------------------------------------------------
id_patterns = [r"subject", r"patient", r"hadm", r"specimen", r"_id$", r"stay_id"]
id_cols = [
    c for c in df.columns
    if any(re.search(p, c, re.IGNORECASE) for p in id_patterns)
]

if id_cols:
    st.info(f"Excluded true IDs from numeric analysis: {', '.join(id_cols)}")

# Convert only real IDs to category – NEVER touch uppercase microbe columns
for c in id_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

# -----------------------------------------------------
# 2. SAFE QUANTITATIVE NUMERIC COLUMNS
# -----------------------------------------------------
numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in id_cols
]

# -----------------------------------------------------
# 3. COLUMN DETECTION
# -----------------------------------------------------
# Microbe columns: ALL UPPERCASE + binary-ish
microbe_cols = [
    c for c in df.columns
    if c.isupper() and df[c].dropna().nunique() <= 3 and pd.api.types.is_numeric_dtype(df[c])
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
# SIDEBAR
# -----------------------------------------------------
module = st.sidebar.radio("Module", [
    "Dataset Explorer", "Microbial Ecology", "Antibiotic Resistance Patterns",
    "Comorbidity Landscape", "Predictive Modeling", "Target Generator (for Maggot Therapy)"
])

# -----------------------------------------------------
# MODULES
# -----------------------------------------------------
if module == "Dataset Explorer":
    st.header("Dataset Explorer")
    if not numeric_cols:
        st.info("No quantitative numeric columns found.")
    else:
        col = st.selectbox("Select variable", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

        st.subheader("Correlation matrix (quantitative only)")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

elif module == "Microbial Ecology":
    st.header("Microbial Ecology")
    if not microbe_cols:
        st.warning("No uppercase binary microbe columns found.")
    else:
        st.write(f"**{len(microbe_cols)}** microbial taxa detected.")
        prev = df[microbe_cols].mean().sort_values(ascending=False)
        st.subheader("Prevalence")
        st.bar_chart(prev.head(30))

        st.subheader("Co-occurrence")
        co = df[microbe_cols].T @ df[microbe_cols]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(co, cmap="mako", ax=ax)
        st.pyplot(fig)

        if len(microbe_cols) >= 2:
            st.subheader("PCA Ecotypes")
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
    if not all([ABX_NAME_COL, ABX_INTERP_COL, microbe_cols]):
        st.warning("Missing abx_names_all / abx_interp_all or microbes.")
    else:
        top_k = st.slider("Antibiotics", 5, 40, 20)
        method = st.radio("Method", ["A: % Resistant", "B: log2 OR", "C: log2 FC"], horizontal=True)
        names, A, B, C = compute_microbe_abx_matrices(df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k)
        if not names:
            st.warning("No resistance data parsed.")
        else:
            heat = A if "A" in method else B if "B" in method else C
            cmap = "viridis" if "A" in method else "coolwarm"
            vmin = 0 if "A" in method else None
            center = None if "A" in method else 0
            fig, ax = plt.subplots(figsize=(14,10))
            sns.heatmap(heat, cmap=cmap, center=center, vmin=vmin, vmax=1 if "A" in method else None, ax=ax)
            ax.set_title(method.split(":")[1])
            st.pyplot(fig)

elif module == "Comorbidity Landscape":
    st.header("Elixhauser Comorbidities")
    if not elix_cols:
        st.warning("No Elixhauser columns.")
    else:
        prev = df[elix_cols].mean().sort_values(ascending=False)
        st.bar_chart(prev)
        if microbe_cols:
            assoc = df[microbe_cols].T @ df[elix_cols]
            fig, ax = plt.subplots(figsize=(12,9))
            sns.heatmap(assoc, cmap="turbo", ax=ax)
            st.pyplot(fig)

elif module == "Predictive Modeling":
    st.header("Predictive Modeling")
    binary_cols = [c for c in df.columns if df[c].nunique() == 2 and c not in id_cols + microbe_cols]
    if not binary_cols:
        st.warning("No binary outcomes found.")
    else:
        target = st.selectbox("Outcome", binary_cols)
        st.checkbox("Microbes", True, key="m1")
        st.checkbox("Comorbidities", True, key="m2")
        st.checkbox("Wound types", False, key="m3")
        st.checkbox("Resistance score", True, key="m4")
        if st.button("Run"):
            run_model(df, target,
                      st.session_state.m1, st.session_state.m2,
                      st.session_state.m3, st.session_state.m4)

elif module == "Target Generator (for Maggot Therapy)":
    st.header("Maggot Therapy Target Generator")
    if not all([ABX_NAME_COL, ABX_INTERP_COL, microbe_cols]):
        st.stop()
    top_k = st.slider("Antibiotics", 5, 40, 20)
    names, A, B, C = compute_microbe_abx_matrices(df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k)
    if not names:
        st.stop()
    prev = df[microbe_cols].mean()
    score = (0.5 * norm_series(A.mean(axis=1)) +
             0.3 * norm_series(B.clip(lower=0).mean(axis=1)) +
             0.2 * norm_series(C.clip(lower=0).mean(axis=1))).fillna(0)
    target_df = pd.DataFrame({"Prevalence": prev, "TargetScore": score}).sort_values("TargetScore", ascending=False)
    st.dataframe(target_df.head(25))
    fig, ax = plt.subplots()
    target_df["TargetScore"].head(20).plot.bar(ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------
# (Keep original helper functions: detect_leakage, run_model, compute_microbe_abx_matrices, norm_series)
# -----------------------------------------------------
# Paste them unchanged from previous working version
