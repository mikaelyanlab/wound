import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wound Ecology Explorer",
                   layout="wide",
                   page_icon="🪰")

# -----------------------------------------------------
# TITLE / INTRO
# -----------------------------------------------------
st.title("🪰 Wound Ecology Explorer")
st.write("""
This tool analyzes wound microbiology, comorbidity structure, and antibiotic resistance 
to identify **microbial targets** relevant for *Lucilia*-based wound biotherapy.
Upload a CSV and explore wound ecotypes, resistance landscapes, 
and predictive models of chronicity or resistance.
""")

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------
uploaded = st.file_uploader("Upload your wound dataset (.csv)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset loaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"**Rows:** {df.shape[0]}  **Columns:** {df.shape[1]}")

    # Identify organism columns
    microbe_cols = [c for c in df.columns if ("_" in c and df[c].nunique() == 2)]
    elix_cols = [c for c in df.columns if c.lower() in
                 ["chf","valv","pulc","perivasc","htn","htnc","para","neuro","diab",
                  "diabwc","hypothy","renal","liver","ulcer","aids","lymph","mets",
                  "tumor","arth","coag","obes","wloss","fed","blane","dane","alcohol",
                  "drug","psycho","depre"]]
    resistance_cols = [c for c in df.columns if "resistance" in c.lower()]

    st.sidebar.header("Navigation")
    module = st.sidebar.radio("Select a module:", [
        "Dataset Explorer",
        "Microbial Ecology",
        "Antibiotic Resistance Patterns",
        "Comorbidity Landscape",
        "Predictive Modeling",
        "Target Generator (for Maggot Therapy)"
    ])

    # -----------------------------------------------------
    # MODULE 1 — DATASET EXPLORER
    # -----------------------------------------------------
    if module == "Dataset Explorer":
        st.header("📊 Dataset Explorer")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.selectbox("Select a numeric variable to visualize", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[selected], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap (numeric variables)")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="viridis", ax=ax2)
        st.pyplot(fig2)

    # -----------------------------------------------------
    # MODULE 2 — MICROBIAL ECOLOGY
    # -----------------------------------------------------
    if module == "Microbial Ecology":
        st.header("🧫 Microbial Ecology")

        st.write("Microbial presence/absence columns detected:", len(microbe_cols))

        # Prevalence
        microbe_prev = df[microbe_cols].mean().sort_values(ascending=False)

        st.subheader("Top Microbes by Prevalence")
        st.bar_chart(microbe_prev.head(25))

        # Co-occurrence heatmap
        st.subheader("Microbial Co-occurrence Network")
        cooccur = df[microbe_cols].T.dot(df[microbe_cols])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cooccur, cmap="magma", xticklabels=False, yticklabels=False)
        st.pyplot(fig)

        # PCA for wound ecotypes
        st.subheader("Wound Ecotypes (PCA Projection)")
        X = df[microbe_cols].values
        pca = PCA(n_components=2).fit_transform(X)
        pca_df = pd.DataFrame(pca, columns=["PC1","PC2"])

        fig3, ax3 = plt.subplots()
        ax3.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.5)
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        st.pyplot(fig3)

    # -----------------------------------------------------
    # MODULE 3 — ANTIBIOTIC RESISTANCE PATTERNS
    # -----------------------------------------------------
    if module == "Antibiotic Resistance Patterns":
        st.header("💊 Antibiotic Resistance Patterns")

        if len(resistance_cols) == 0:
            st.warning("No resistance columns detected.")
        else:
            st.subheader("Distribution of Resistance Score")
            for col in resistance_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)

            st.subheader("Resistance × Microbe Heatmap")
            microbial_means = df[microbe_cols].T.dot(df[resistance_cols])
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.heatmap(microbial_means, cmap="coolwarm")
            st.pyplot(fig2)

    # -----------------------------------------------------
    # MODULE 4 — COMORBIDITY LANDSCAPE
    # -----------------------------------------------------
    if module == "Comorbidity Landscape":
        st.header("⚕️ Elixhauser Comorbidity Landscape")

        comorb_counts = df[elix_cols].mean().sort_values(ascending=False)
        st.subheader("Prevalence of Comorbidities")
        st.bar_chart(comorb_counts)

        st.subheader("Microbe–Comorbidity Associations")
        assoc = df[microbe_cols].T.dot(df[elix_cols])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(assoc, cmap="turbo")
        st.pyplot(fig)

    # -----------------------------------------------------
    # MODULE 5 — PREDICTIVE MODELING
    # -----------------------------------------------------
    if module == "Predictive Modeling":
        st.header("🤖 Predictive Modeling")

        st.write("Choose an outcome variable (0/1) to model:")
        target = st.selectbox("Outcome", [c for c in df.columns if df[c].nunique() == 2])

        # Features = microbes + comorbidities + resistance
        X = df[microbe_cols + elix_cols + resistance_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        st.subheader(f"Model AUC: {auc:.3f}")

        feature_imp = pd.Series(model.feature_importances_, index=X.columns)
        st.subheader("Top Predictive Features")
        st.bar_chart(feature_imp.sort_values(ascending=False).head(25))

    # -----------------------------------------------------
    # MODULE 6 — TARGET GENERATOR (MAGGOT THERAPY)
    # -----------------------------------------------------
    if module == "Target Generator (for Maggot Therapy)":
        st.header("🎯 Maggot Therapy Target Generator")

        st.write("""
        This module identifies microbes most associated with:
        - chronicity  
        - multidrug resistance  
        - comorbidity burden  
        These represent *high-value microbial targets* for experimental maggot assays.
        """)

        # 1. Microbes enriched in chronic wounds
        if "chronic" in df.columns:
            chronic_assoc = df.groupby("chronic")[microbe_cols].mean().T
            chronic_assoc["delta"] = chronic_assoc[1] - chronic_assoc[0]
            st.subheader("Microbes Enriched in Chronic Wounds")
            st.dataframe(chronic_assoc.sort_values("delta", ascending=False).head(20))

        # 2. Microbes linked to high resistance
        if len(resistance_cols):
            resistance_signal = df[resistance_cols].mean(axis=1)
            corr = df[microbe_cols].corrwith(resistance_signal).sort_values(ascending=False)
            st.subheader("Microbes Associated with High Antibiotic Resistance")
            st.dataframe(corr.head(20))

        # 3. Microbes enriched in high-comorbidity patients
        comorb_sum = df[elix_cols].sum(axis=1)
        corr_comorb = df[microbe_cols].corrwith(comorb_sum).sort_values(ascending=False)

        st.subheader("Microbes Associated with High Comorbidity Burden")
        st.dataframe(corr_comorb.head(20))

        st.markdown("---")
        st.success("These lists can be combined to form a **Tier 1 microbial target panel** for Lucilia assays.")
