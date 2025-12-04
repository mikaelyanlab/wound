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
st.set_page_config(page_title="Wound Ecology Explorer", layout="wide", page_icon="fly")

st.title("Wound Ecology Explorer")
st.write("""
This app explores wound microbiology, comorbidities, and antibiotic resistance  
to help identify **microbial targets** for *Lucilia*-based wound biotherapy.  
Upload a CSV and use the modules in the sidebar.
""")

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
# EXCLUDE ID-LIKE COLUMNS (CRITICAL FIX)
# -----------------------------------------------------
id_patterns = ["id", "ID", "Id", "subject", "patient", "hadm", "specimen", "stay"]
id_cols = [c for c in df.columns if any(p in c for p in id_patterns)]

if id_cols:
    st.info(f"Excluded identifier columns from numeric analysis: {', '.join(id_cols)}")
    # Convert to category so pandas excludes them from numeric ops
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

# True quantitative numeric columns
numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in id_cols
]

# -----------------------------------------------------
# COLUMN GROUP DETECTION
# -----------------------------------------------------
microbe_cols = [
    c for c in df.columns
    if c.isupper() and df[c].dropna().nunique() <= 3
]

ELIX_COLS = ["chf","valv","pulc","perivasc","htn","htnc","para","neuro",
             "diab","diabwc","hypothy","renal","liver","ulcer","aids",
             "lymph","mets","tumor","arth","coag","obes","wloss","fed",
             "blane","dane","alcohol","drug","psycho","depre"]
elix_cols = [c for c in ELIX_COLS if c in df.columns]

wound_type_vars = [c for c in df.columns if c.startswith("wound_")]

ABX_NAME_COL = "abx_names_all" if "abx_names_all" in df.columns else None
ABX_INTERP_COL = "abx_interp_all" if "abx_interp_all" in df.columns else None

resistance_score_cols = [c for c in df.columns if "resistance_score" in c]

# -----------------------------------------------------
# HELPER FUNCTIONS (unchanged except leakage detection now safer)
# -----------------------------------------------------
def detect_leakage(X, y):
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

    feature_cols = [c for c in feature_cols if c not in [target_col] + id_cols]
    forbidden = OUTCOME_DEPENDENT_BLOCKLIST.get(target_col, [])
    feature_cols = [c for c in feature_cols if c not in forbidden]

    if not feature_cols:
        st.error("No predictor variables selected.")
        return

    X = df[feature_cols].astype(float)

    leakage = detect_leakage(X, y)
    if leakage:
        st.warning(f"Potential leakage detected – removing: {leakage}")
        X = X.drop(columns=leakage)
        feature_cols = [c for c in feature_cols if c not in leakage]

    if X.shape[1] == 0:
        st.error("No predictors remain after filtering.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=400, max_depth=10,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    st.subheader(f"Model AUC: {auc:.3f}")
    if np.isclose(auc, 1.0):
        st.warning("Perfect AUC – possible remaining leakage.")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(25)
    if importances.iloc[0] > 0.5:
        st.warning(f"One feature dominates importance: {importances.index[0]}")

    fig, ax = plt.subplots(figsize=(12, 4))
    importances.plot(kind="bar", ax=ax)
    ax.set_title("Top Predictive Features")
    st.pyplot(fig)

# -----------------------------------------------------
# Microbe × Antibiotic matrices (unchanged)
# -----------------------------------------------------
def compute_microbe_abx_matrices(df, microbe_cols, abx_name_col, abx_interp_col, top_k=20):
    # ... (exact same function as original – omitted for brevity) ...
    # (Copy-paste the original function here – it is unchanged)
    abx_total_N = defaultdict(int)
    abx_total_R = defaultdict(int)
    N_present = defaultdict(int)
    R_present = defaultdict(int)
    for _, row in df.iterrows():
        names = row.get(abx_name_col)
        interps = row.get(abx_interp_col)
        if not isinstance(names, str) or not isinstance(interps, str):
            continue
        names_list = [n.strip().upper() for n in names.split(",") if n.strip()]
        interps_list = [i.strip().upper() for i in interps.split(",") if i.strip()]
        if not names_list:
            continue
        L = min(len(names_list), len(interps_list))
        names_list = names_list[:L]
        interps_list = interps_list[:L]
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
    abx_sorted = sorted(abx_total_N.items(), key=lambda x: x[1], reverse=True)
    abx_names = [a for a, _ in abx_sorted[:top_k]] if top_k else [a for a, _ in abx_sorted]
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
            if n_present > 0:
                heat_A.loc[m, a] = r_present / n_present
            n_absent = n_total - n_present
            r_absent = r_total - r_present
            a_ct = r_present + 0.5
            b_ct = (n_present - r_present) + 0.5
            c_ct = r_absent + 0.5
            d_ct = (n_absent - r_absent) + 0.5
            OR = (a_ct / b_ct) / (c_ct / d_ct)
            heat_B_log2.loc[m, a] = math.log2(OR) if OR > 0 else np.nan
            p1 = r_present / n_present if n_present > 0 else None
            p0 = r_absent / n_absent if n_absent > 0 else None
            if p1 is None or p0 is None or p1 == 0 or p0 == 0:
                p1 = p1 if (p1 is not None and p1 > 0) else 1e-6
                p0 = p0 if (p0 is not None and p0 > 0) else 1e-6
                heat_C.loc[m, a] = math.log2(p1 / p0)
            else:
                heat_C.loc[m, a] = math.log2(p1 / p0)
    return abx_names, heat_A, heat_B_log2, heat_C

def norm_series(s):
    s = s.copy()
    if s.dropna().empty:
        return s
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
module = st.sidebar.radio("Select a module:", [
    "Dataset Explorer",
    "Microbial Ecology",
    "Antibiotic Resistance Patterns",
    "Comorbidity Landscape",
    "Predictive Modeling",
    "Target Generator (for Maggot Therapy)"
])

# -----------------------------------------------------
# MODULES (updated where needed)
# -----------------------------------------------------
if module == "Dataset Explorer":
    st.header("Dataset Explorer")
    if numeric_cols:
        chosen = st.selectbox("Quantitative variable", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[chosen], kde=True, ax=ax)
        ax.set_title(f"Distribution of {chosen}")
        st.pyplot(fig)

        st.subheader("Correlation heatmap (quantitative variables only)")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="viridis", ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("No quantitative numeric columns found (ID columns excluded).")

elif module == "Microbial Ecology":
    st.header("Microbial Ecology")
    if not microbe_cols:
        st.warning("No microbe columns detected.")
    else:
        st.write(f"Detected **{len(microbe_cols)}** microbial columns.")
        prev = df[microbe_cols].mean().sort_values(ascending=False)
        st.subheader("Top Microbes by Prevalence")
        st.bar_chart(prev.head(25))

        st.subheader("Co-occurrence Heatmap")
        co = df[microbe_cols].T.dot(df[microbe_cols])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(co, cmap="magma", xticklabels=False, yticklabels=False, ax=ax)
        st.pyplot(fig)

        if len(microbe_cols) >= 2:
            st.subheader("Wound Ecotypes (PCA)")
            X = df[microbe_cols].fillna(0)
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            fig2, ax2 = plt.subplots()
            ax2.scatter(pcs[:, 0], pcs[:, 1], alpha=0.5, s=10)
            ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            st.pyplot(fig2)

elif module == "Antibiotic Resistance Patterns":
    st.header("Antibiotic Resistance Patterns")
    if not (ABX_NAME_COL and ABX_INTERP_COL and microbe_cols):
        st.warning("Missing required columns.")
    else:
        top_k = st.slider("Antibiotics to show", 5, 40, 20)
        method = st.radio("Method", ["A: Proportion Resistant", "B: Odds Ratio", "C: Log2 Fold-Change"], horizontal=True)
        abx_names, heat_A, heat_B_log2, heat_C = compute_microbe_abx_matrices(
            df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k=top_k)
        if not abx_names:
            st.warning("No resistance data.")
        else:
            heat = heat_A if method.startswith("A") else heat_B_log2 if method.startswith("B") else heat_C
            title = method.split(":")[1].strip()
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(heat, cmap="viridis" if method.startswith("A") else "coolwarm",
                        center=0 if not method.startswith("A") else None,
                        vmin=0, vmax=1 if method.startswith("A") else None,
                        ax=ax)
            ax.set_title(title)
            st.pyplot(fig)

elif module == "Comorbidity Landscape":
    st.header("Elixhauser Comorbidity Landscape")
    if not elix_cols:
        st.warning("No Elixhauser columns.")
    else:
        prev = df[elix_cols].mean().sort_values(ascending=False)
        st.bar_chart(prev)
        if microbe_cols:
            assoc = df[microbe_cols].T.dot(df[elix_cols])
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(assoc, cmap="turbo", ax=ax)
            ax.set_xlabel("Comorbidity")
            ax.set_ylabel("Microbe")
            st.pyplot(fig)

elif module == "Predictive Modeling":
    st.header("Predictive Modeling")
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2 and c not in microbe_cols + id_cols]
    if not binary_cols:
        st.warning("No suitable binary outcomes.")
    else:
        target_col = st.selectbox("Outcome", binary_cols)
        st.markdown("**Features to include**")
        use_microbes = st.checkbox("Microbes", True)
        use_comorb = st.checkbox("Comorbidities", True)
        use_wound_types = st.checkbox("Wound-type vars", False)
        use_resistance_score = st.checkbox("Resistance score", True)
        if st.button("Run model"):
            run_model(df, target_col,
                      use_microbes=use_microbes,
                      use_comorb=use_comorb,
                      use_wound_types=use_wound_types,
                      use_resistance_score=use_resistance_score)

elif module == "Target Generator (for Maggot Therapy)":
    st.header("Maggot Therapy Target Generator")
    if not (ABX_NAME_COL and ABX_INTERP_COL and microbe_cols):
        st.warning("Missing required columns.")
        st.stop()
    top_k = st.slider("Antibiotics for scoring", 5, 40, 20)
    abx_names, heat_A, heat_B_log2, heat_C = compute_microbe_abx_matrices(
        df, microbe_cols, ABX_NAME_COL, ABX_INTERP_COL, top_k=top_k)
    if not abx_names:
        st.stop()

    A = heat_A.mean(axis=1)
    B = heat_B_log2.clip(lower=0).mean(axis=1)
    C = heat_C.clip(lower=0).mean(axis=1)
    prev = df[microbe_cols].mean()

    score = (0.5 * norm_series(A) + 0.3 * norm_series(B) + 0.2 * norm_series(C)).fillna(0)
    target_df = pd.DataFrame({
        "Prevalence": prev,
        "Mean % Resistant": A,
        "Mean log2 OR (pos)": B,
        "Mean log2 FC (pos)": C,
        "TargetScore": score
    }).sort_values("TargetScore", ascending=False)

    st.dataframe(target_df.head(25))
    fig, ax = plt.subplots(figsize=(10, 6))
    target_df["TargetScore"].head(20).plot(kind="bar", ax=ax)
    ax.set_ylabel("Target Score")
    plt.xticks(rotation=75, ha="right")
    st.pyplot(fig)
