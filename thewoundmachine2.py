import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Wound Ecology Explorer",
    layout="wide",
    page_icon="🪰"
)

st.title("🪰 Wound Ecology Explorer")
st.write(
    """
This app explores wound microbiology, comorbidities, and antibiotic resistance
to help identify **microbial targets** for *Lucilia*-based wound biotherapy.
Upload a CSV and use the modules in the sidebar to explore.
"""
)

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------
uploaded = st.file_uploader("Upload your wound dataset (.csv)", type="csv")

if uploaded is None:
    st.info("Upload the wound CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.write(f"**Rows:** {df.shape[0]}  **Columns:** {df.shape[1]}")

# -----------------------------------------------------
# COLUMN GROUP DETECTION
# -----------------------------------------------------

# 1. Microbes: ALL UPPERCASE column names, binary (0/1)
microbe_cols = [
    c for c in df.columns
    if c.isupper() and df[c].dropna().nunique() <= 3
]

# 2. Elixhauser comorbidities (explicit list to avoid surprises)
ELIX_COLS = [
    "chf", "valv", "pulc", "perivasc", "htn", "htnc", "para", "neuro",
    "diab", "diabwc", "hypothy", "renal", "liver", "ulcer", "aids",
    "lymph", "mets", "tumor", "arth", "coag", "obes", "wloss", "fed",
    "blane", "dane", "alcohol", "drug", "psycho", "depre"
]
elix_cols = [c for c in ELIX_COLS if c in df.columns]

# 3. Wound-type variables
wound_type_vars = [c for c in df.columns if c.startswith("wound_")]

# 4. Antibiotic interpretation columns (per-antibiotic)
antibiotic_cols = [
    c for c in df.columns
    if any(k in c.lower() for k in ["abx", "antibiotic", "susc", "interp"])
]

# Resistance score column (numeric aggregate)
resistance_score_cols = [c for c in df.columns if "resistance_score" in c]

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def encode_resistance(val):
    """Convert S/I/R / text interpretations to 0/1/NaN."""
    if isinstance(val, str):
        v = val.strip().upper()
        if v in ["R", "RESISTANT"]:
            return 1
        if v in ["S", "SUSCEPTIBLE"]:
            return 0
        if v in ["I", "INTERMEDIATE"]:
            # could be coded as 0.5 but for now treat as resistant-ish
            return 1
    return np.nan

def detect_leakage(X, y):
    """Flag features that are >0.95 correlated with the outcome."""
    leakage_features = []
    y_arr = y.values.astype(float)
    if y_arr.std() == 0:
        return leakage_features
    for col in X.columns:
        x = X[col].values.astype(float)
        if x.std() == 0:
            continue
        corr = np.corrcoef(x, y_arr)[0, 1]
        if np.isfinite(corr) and abs(corr) > 0.95:
            leakage_features.append(col)
    return leakage_features

# Outcomes where certain predictors should be excluded
OUTCOME_DEPENDENT_BLOCKLIST = {
    "infection_persistent_any_site": wound_type_vars,
    "infection_persistent_same_site": wound_type_vars,
    "multi_drug_resistant": wound_type_vars + resistance_score_cols,
    "wound_burn": wound_type_vars,
    "wound_open": wound_type_vars,
    "wound_ulcer": wound_type_vars,
    "wound_surgical": wound_type_vars,
}

def run_model(df, target_col, use_microbes=True, use_comorb=True,
              use_wound_types=False, use_resistance_score=True):
    """Leakage-aware RF model."""

    # Clean outcome
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    if y.dropna().nunique() != 2:
        st.error("Selected outcome is not binary.")
        return

    y = y.astype(int)

    feature_cols = []

    if use_microbes and microbe_cols:
        feature_cols.extend(microbe_cols)

    if use_comorb and elix_cols:
        feature_cols.extend(elix_cols)

    if use_wound_types and wound_type_vars:
        feature_cols.extend(wound_type_vars)

    if use_resistance_score and resistance_score_cols:
        feature_cols.extend(resistance_score_cols)

    # always drop the target if somehow in features
    feature_cols = [c for c in feature_cols if c != target_col]

    # remove outcome-dependent forbidden vars
    forbidden = OUTCOME_DEPENDENT_BLOCKLIST.get(target_col, [])
    feature_cols = [c for c in feature_cols if c not in forbidden]

    X = df[feature_cols].astype(float)

    # leakage detection
    leakage = detect_leakage(X, y)
    if leakage:
        st.warning(
            f"⚠ Potential information leakage detected; "
            f"removing features: {leakage}"
        )
        X = X.drop(columns=leakage)
        feature_cols = [c for c in feature_cols if c not in leakage]

    if X.shape[1] == 0:
        st.error("No predictor variables left after leakage filtering.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        st.error("Stratified split failed due to extreme class imbalance.")
        return

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    st.subheader(f"Model AUC: {auc:.3f}")

    if np.isclose(auc, 1.0):
        st.warning(
            "AUC is 1.0 — this often indicates leakage or a trivially separable outcome."
        )

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(25)

    if importances.iloc[0] > 0.5:
        st.warning(
            f"⚠ One feature ({importances.index[0]}) accounts for "
            f"{importances.iloc[0]*100:.1f}% of model importance. "
            "Interpret with caution."
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    importances.plot(kind="bar", ax=ax)
    ax.set_title("Top Predictive Features")
    ax.set_ylabel("Importance")
    st.pyplot(fig)

    return model

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.header("Navigation")
module = st.sidebar.radio(
    "Select a module:",
    [
        "Dataset Explorer",
        "Microbial Ecology",
        "Antibiotic Resistance Patterns",
        "Comorbidity Landscape",
        "Predictive Modeling",
        "Target Generator (for Maggot Therapy)"
    ]
)

# -----------------------------------------------------
# MODULE: DATASET EXPLORER
# -----------------------------------------------------
if module == "Dataset Explorer":
    st.header("📊 Dataset Explorer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        chosen = st.selectbox("Numeric variable to visualize", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[chosen], kde=True, ax=ax)
        ax.set_title(f"Distribution of {chosen}")
        st.pyplot(fig)

        st.subheader("Correlation heatmap (numeric variables)")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="viridis")
        st.pyplot(fig2)
    else:
        st.info("No numeric columns detected.")

# -----------------------------------------------------
# MODULE: MICROBIAL ECOLOGY
# -----------------------------------------------------
elif module == "Microbial Ecology":
    st.header("🧫 Microbial Ecology")

    if not microbe_cols:
        st.warning("No microbe columns detected (ALL UPPERCASE).")
    else:
        st.write(f"Detected **{len(microbe_cols)}** microbial presence columns.")

        # prevalence
        micro_prev = df[microbe_cols].mean().sort_values(ascending=False)
        st.subheader("Top Microbes by Prevalence")
        st.bar_chart(micro_prev.head(25))

        # co-occurrence
        st.subheader("Microbial Co-occurrence Heatmap")
        cooccur = df[microbe_cols].T.dot(df[microbe_cols])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cooccur, cmap="magma", xticklabels=False, yticklabels=False)
        st.pyplot(fig)

        # PCA projection
        st.subheader("Wound Ecotypes (PCA on Microbes)")
        X = df[microbe_cols].fillna(0).values
        if X.shape[1] >= 2:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            fig2, ax2 = plt.subplots()
            ax2.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.4, s=10)
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            st.pyplot(fig2)
        else:
            st.info("Not enough microbial dimensions for PCA.")

# -----------------------------------------------------
# MODULE: ANTIBIOTIC RESISTANCE PATTERNS
# -----------------------------------------------------
elif module == "Antibiotic Resistance Patterns":
    st.header("💊 Antibiotic Resistance Patterns")

    # show resistance_score distribution if present
    if resistance_score_cols:
        for col in resistance_score_cols:
            st.subheader(f"Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # per-antibiotic interpretation
    if not antibiotic_cols:
        st.warning("No antibiotic interpretation columns detected.")
    else:
        st.write(f"Detected **{len(antibiotic_cols)}** antibiotic-related columns.")

        for col in antibiotic_cols:
            bin_col = col + "_bin"
            df[bin_col] = df[col].apply(encode_resistance)

        binary_abx_cols = [c for c in df.columns if c.endswith("_bin")]

        if microbe_cols and binary_abx_cols:
            st.subheader("Microbe × Antibiotic Resistance Heatmap")

            # counts of resistance per microbe–antibiotic
            heat_raw = df[microbe_cols].T.dot(df[binary_abx_cols])

            # normalize by microbe prevalence (proportion resistant)
            micro_counts = df[microbe_cols].sum(axis=0).replace(0, np.nan)
            heat_norm = heat_raw.div(micro_counts, axis=0)

            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                heat_norm,
                cmap="viridis",
                xticklabels=[c.replace("_bin", "") for c in binary_abx_cols],
                yticklabels=microbe_cols
            )
            ax.set_xlabel("Antibiotic (resistant proportion)")
            ax.set_ylabel("Microbe")
            st.pyplot(fig)
        else:
            st.info("Need both microbe and antibiotic columns for this heatmap.")

# -----------------------------------------------------
# MODULE: COMORBIDITY LANDSCAPE
# -----------------------------------------------------
elif module == "Comorbidity Landscape":
    st.header("⚕️ Elixhauser Comorbidity Landscape")

    if not elix_cols:
        st.warning("No Elixhauser comorbidity columns detected.")
    else:
        comorb_prev = df[elix_cols].mean().sort_values(ascending=False)
        st.subheader("Prevalence of Comorbidities")
        st.bar_chart(comorb_prev)

        if microbe_cols:
            st.subheader("Microbe–Comorbidity Associations")
            assoc = df[microbe_cols].T.dot(df[elix_cols])
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                assoc,
                cmap="turbo",
                xticklabels=elix_cols,
                yticklabels=microbe_cols
            )
            ax.set_xlabel("Comorbidity")
            ax.set_ylabel("Microbe")
            st.pyplot(fig)
        else:
            st.info("Microbe columns not detected; cannot compute associations.")

# -----------------------------------------------------
# MODULE: PREDICTIVE MODELING
# -----------------------------------------------------
elif module == "Predictive Modeling":
    st.header("🤖 Predictive Modeling")

    # binary outcomes only
    binary_cols = [
        c for c in df.columns
        if df[c].dropna().nunique() == 2 and c not in microbe_cols
    ]

    if not binary_cols:
        st.warning("No binary outcome columns found.")
    else:
        target_col = st.selectbox("Select outcome variable (binary)", binary_cols)

        st.markdown("**Feature groups to include:**")
        use_microbes = st.checkbox("Microbes", value=True)
        use_comorb = st.checkbox("Comorbidities", value=True)
        use_wound_types = st.checkbox("Wound-type variables", value=False)
        use_resistance_score = st.checkbox("Resistance score", value=True)

        if st.button("Run model"):
            run_model(
                df,
                target_col,
                use_microbes=use_microbes,
                use_comorb=use_comorb,
                use_wound_types=use_wound_types,
                use_resistance_score=use_resistance_score
            )

# -----------------------------------------------------
# MODULE: TARGET GENERATOR
# -----------------------------------------------------
elif module == "Target Generator (for Maggot Therapy)":
    st.header("🎯 Maggot Therapy Target Generator")

    if not microbe_cols:
        st.warning("No microbe columns detected.")
        st.stop()

    st.write(
        """
This module flags microbes that are:
- enriched in **chronic or persistent** infections,
- associated with **high resistance**, and
- associated with **high comorbidity burden**.

These become **Tier 1 targets** for *Lucilia* assays.
"""
    )

    # 1. Chronicity enrichment, if a suitable column exists
    chronic_candidates = [
        c for c in df.columns
        if "chronic" in c.lower()
        or "persistent" in c.lower()
        or "infection_persistent_any_site" == c
    ]
    if chronic_candidates:
        chronic_col = chronic_candidates[0]
        st.subheader(f"Microbes enriched in chronic/persistent wounds ({chronic_col})")
        tmp = df.dropna(subset=[chronic_col])
        grp = tmp.groupby(tmp[chronic_col])[microbe_cols].mean().T
        if 1 in grp.columns and 0 in grp.columns:
            grp["delta"] = grp[1] - grp[0]
            st.dataframe(grp.sort_values("delta", ascending=False).head(25))
        else:
            st.info("Chronicity column is not coded as 0/1.")

    # 2. Association with resistance_score
    if resistance_score_cols:
        st.subheader("Microbes associated with high resistance_score")
        rs = df[resistance_score_cols[0]]
        corr_rs = df[microbe_cols].corrwith(rs).sort_values(ascending=False)
        st.dataframe(corr_rs.head(25))

    # 3. Association with comorbidity burden
    if elix_cols:
        st.subheader("Microbes associated with high comorbidity burden")
        comorb_sum = df[elix_cols].sum(axis=1)
        corr_comorb = df[microbe_cols].corrwith(comorb_sum).sort_values(ascending=False)
        st.dataframe(corr_comorb.head(25))

    st.markdown(
        "---\n"
        "Combine these lists into a **Tier 1 microbial panel** for experimental "
        "testing with *Lucilia* secretions, symbionts, or engineered consortia."
    )
