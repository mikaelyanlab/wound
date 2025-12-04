import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from collections import defaultdict
import math

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

# 1. Microbes: ALL UPPERCASE column names, roughly binary (0/1)
microbe_cols = [
    c for c in df.columns
    if c.isupper() and df[c].dropna().nunique() <= 3
]

# 2. Elixhauser comorbidities (subset if present)
ELIX_COLS = [
    "chf", "valv", "pulc", "perivasc", "htn", "htnc", "para", "neuro",
    "diab", "diabwc", "hypothy", "renal", "liver", "ulcer", "aids",
    "lymph", "mets", "tumor", "arth", "coag", "obes", "wloss", "fed",
    "blane", "dane", "alcohol", "drug", "psycho", "depre"
]
elix_cols = [c for c in ELIX_COLS if c in df.columns]

# 3. Wound-type variables
wound_type_vars = [c for c in df.columns if c.startswith("wound_")]

# 4. Antibiotic summary columns
ABX_NAME_COL = "abx_names_all" if "abx_names_all" in df.columns else None
ABX_INTERP_COL = "abx_interp_all" if "abx_interp_all" in df.columns else None

# 5. Resistance score column (numeric aggregate)
resistance_score_cols = [c for c in df.columns if "resistance_score" in c]

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------

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

    feature_cols = [c for c in feature_cols if c != target_col]

    forbidden = OUTCOME_DEPENDENT_BLOCKLIST.get(target_col, [])
    feature_cols = [c for c in feature_cols if c not in forbidden]

    if not feature_cols:
        st.error("No predictor variables selected for modeling.")
        return

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

def compute_microbe_abx_matrices(df, microbe_cols, abx_name_col, abx_interp_col, top_k=20):
    """
    Build microbe × antibiotic matrices for:
      - A: proportion resistant
      - B: log2 odds ratio
      - C: log2 fold-change in resistance probability
    using abx_names_all + abx_interp_all.
    """
    abx_total_N = defaultdict(int)  # how many times each abx was tested
    abx_total_R = defaultdict(int)  # how many of those were resistant
    N_present = defaultdict(int)    # (microbe, abx): count where microbe present & abx tested
    R_present = defaultdict(int)    # (microbe, abx): count where microbe present & abx resistant

    for _, row in df.iterrows():
        names = row.get(abx_name_col)
        interps = row.get(abx_interp_col)
        if not isinstance(names, str) or not isinstance(interps, str):
            continue

        names_list = [n.strip().upper() for n in names.split(",") if n.strip()]
        interps_list = [i.strip().upper() for i in interps.split(",") if i.strip()]
        if not names_list or not interps_list:
            continue

        L = min(len(names_list), len(interps_list))
        names_list = names_list[:L]
        interps_list = interps_list[:L]

        # microbes present in this row
        micro_present = [m for m in microbe_cols if row[m] == 1]

        for name, interp in zip(names_list, interps_list):
            abx_total_N[name] += 1
            is_res = 1 if interp in ("R", "RESISTANT", "I", "INTERMEDIATE") else 0
            if is_res:
                abx_total_R[name] += 1

            for m in micro_present:
                N_present[(m, name)] += 1
                if is_res:
                    R_present[(m, name)] += 1

    if not abx_total_N:
        return [], None, None, None

    # pick most commonly used antibiotics
    abx_sorted = sorted(abx_total_N.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        abx_names = [a for a, _ in abx_sorted[:top_k]]
    else:
        abx_names = [a for a, _ in abx_sorted]

    idx = list(microbe_cols)
    cols = list(abx_names)

    heat_A = pd.DataFrame(np.nan, index=idx, columns=cols)
    heat_B_log2 = pd.DataFrame(np.nan, index=idx, columns=cols)
    heat_C = pd.DataFrame(np.nan, index=idx, columns=cols)

    for m in idx:
        for a in cols:
            n_present = N_present.get((m, a), 0)
            r_present = R_present.get((m, a), 0)
            n_total = abx_total_N.get(a, 0)
            r_total = abx_total_R.get(a, 0)

            # Method A: proportion resistant, given microbe present
            if n_present > 0:
                heat_A.loc[m, a] = r_present / n_present

            # Derived counts for absent case
            n_absent = n_total - n_present
            r_absent = r_total - r_present

            # Method B: odds ratio with Haldane correction
            a_ct = r_present + 0.5
            b_ct = (n_present - r_present) + 0.5
            c_ct = r_absent + 0.5
            d_ct = (n_absent - r_absent) + 0.5
            OR = (a_ct / b_ct) / (c_ct / d_ct)
            heat_B_log2.loc[m, a] = math.log2(OR) if OR > 0 else np.nan

            # Method C: log2 fold-change in P(resistance | microbe present vs absent)
            p1 = r_present / n_present if n_present > 0 else None
            p0 = r_absent / n_absent if n_absent > 0 else None

            if p1 is None or p0 is None or p1 == 0 or p0 == 0:
                if (p1 is not None and p1 > 0) or (p0 is not None and p0 > 0):
                    p1 = p1 if (p1 is not None and p1 > 0) else 1e-6
                    p0 = p0 if (p0 is not None and p0 > 0) else 1e-6
                    heat_C.loc[m, a] = math.log2(p1 / p0)
                else:
                    heat_C.loc[m, a] = np.nan
            else:
                heat_C.loc[m, a] = math.log2(p1 / p0)

    return abx_names, heat_A, heat_B_log2, heat_C

def norm_series(s):
    """Min–max normalize a Series, keeping NaNs."""
    s = s.copy()
    if s.dropna().empty:
        return s
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

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
# MODULE: ANTIBIOTIC RESISTANCE PATTERNS (A, B, C)
# -----------------------------------------------------
elif module == "Antibiotic Resistance Patterns":
    st.header("💊 Antibiotic Resistance Patterns")

    if ABX_NAME_COL is None or ABX_INTERP_COL is None:
        st.warning("No `abx_names_all` / `abx_interp_all` columns detected.")
    elif not microbe_cols:
        st.warning("No microbe columns detected (ALL UPPERCASE).")
    else:
        top_k = st.slider("Number of antibiotics to display", 5, 40, 20)

        method = st.radio(
            "Select resistance–association method:",
            ["A: Proportion Resistant", "B: Odds Ratio Enrichment", "C: Log2 Fold-Change"],
            horizontal=True
        )

        abx_names, heat_A, heat_B_log2, heat_C = compute_microbe_abx_matrices(
            df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k=top_k
        )

        if not abx_names or heat_A is None:
            st.warning("Unable to compute antibiotic statistics from the dataset.")
        else:
            if method.startswith("A"):
                st.subheader("Method A: Proportion Resistant Given Microbe Presence")
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(
                    heat_A,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    xticklabels=abx_names,
                    yticklabels=microbe_cols
                )
                ax.set_title("P(resistant | microbe present)")
                ax.set_xlabel("Antibiotic")
                ax.set_ylabel("Microbe")
                st.pyplot(fig)

            elif method.startswith("B"):
                st.subheader("Method B: Odds Ratio Enrichment (log2 scale)")
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(
                    heat_B_log2,
                    cmap="coolwarm",
                    center=0,
                    xticklabels=abx_names,
                    yticklabels=microbe_cols
                )
                ax.set_title("log2(OR) of Resistance Given Microbe Presence")
                ax.set_xlabel("Antibiotic")
                ax.set_ylabel("Microbe")
                st.pyplot(fig)

            elif method.startswith("C"):
                st.subheader("Method C: Log2 Fold-Change in Resistance Probability")
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(
                    heat_C,
                    cmap="coolwarm",
                    center=0,
                    xticklabels=abx_names,
                    yticklabels=microbe_cols
                )
                ax.set_title("log2[P(res | M=1) / P(res | M=0)]")
                ax.set_xlabel("Antibiotic")
                ax.set_ylabel("Microbe")
                st.pyplot(fig)

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
# MODULE: TARGET GENERATOR (Maggot Therapy)
# -----------------------------------------------------
elif module == "Target Generator (for Maggot Therapy)":
    st.header("🎯 Maggot Therapy Target Generator")

    if ABX_NAME_COL is None or ABX_INTERP_COL is None:
        st.warning("No `abx_names_all` / `abx_interp_all` columns detected.")
        st.stop()
    if not microbe_cols:
        st.warning("No microbe columns detected (ALL UPPERCASE).")
        st.stop()

    st.write(
        """
This module ranks microbes as **targets** for *Lucilia* assays using:
- Method A: baseline resistance burden,
- Method B: enrichment relative to background,
- Method C: ecological shift in resistance,
plus **prevalence**.
"""
    )

    top_k = st.slider("Number of antibiotics to use for scoring", 5, 40, 20)

    abx_names, heat_A, heat_B_log2, heat_C = compute_microbe_abx_matrices(
        df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k=top_k
    )

    if not abx_names or heat_A is None:
        st.warning("Unable to compute antibiotic statistics from the dataset.")
        st.stop()

    # Per-microbe summaries
    A_raw = heat_A.mean(axis=1, skipna=True)                   # baseline burden
    B_raw = heat_B_log2.clip(lower=0).mean(axis=1, skipna=True)  # positive enrichment only
    C_raw = heat_C.clip(lower=0).mean(axis=1, skipna=True)        # positive ecological shifts

    # Prevalence of each microbe
    prevalence = df[microbe_cols].mean()

    # Normalize components
    A_norm = norm_series(A_raw)
    B_norm = norm_series(B_raw)
    C_norm = norm_series(C_raw)
    prev_norm = norm_series(prevalence)

    # Target score: combine components (weights can be tuned)
    target_score = (
        0.5 * A_norm.fillna(0) +
        0.3 * B_norm.fillna(0) +
        0.2 * C_norm.fillna(0)
    )

    target_df = pd.DataFrame({
        "prevalence": prevalence,
        "A_mean_prop_resistant": A_raw,
        "B_mean_log2_OR_pos": B_raw,
        "C_mean_log2_fold_pos": C_raw,
        "TargetScore": target_score
    }).sort_values("TargetScore", ascending=False)

    st.subheader("Ranked Microbial Targets")
    st.dataframe(target_df.head(25))

    st.subheader("Top Targets by Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    target_df["TargetScore"].head(20).plot(kind="bar", ax=ax)
    ax.set_ylabel("Target Score (normalized)")
    ax.set_xlabel("Microbe")
    plt.xticks(rotation=75, ha="right")
    st.pyplot(fig)

    st.markdown(
        """
**Interpretation:**
- High **TargetScore** + high **prevalence** → *Tier 1* candidates for *Lucilia* assays.
- High **A** but low **B/C** → common but not disproportionate drivers.
- High **B/C** but low prevalence → niche but *interesting* ecological disruptors.
"""
    )
