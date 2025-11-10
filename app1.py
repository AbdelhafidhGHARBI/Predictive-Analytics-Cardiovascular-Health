# app.py
# Streamlit dashboard — Patient Health ML (EDA, Encoding, PCA, Baselines)
# ---------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Optional libs
try:
    from ydata_profiling import ProfileReport
    YDATA_OK = True
except Exception:
    YDATA_OK = False

try:
    import umap.umap_ as umap
    UMAP_OK = True
except Exception:
    UMAP_OK = False

try:
    from sklearn.manifold import TSNE
    TSNE_OK = True
except Exception:
    TSNE_OK = False

try:
    import category_encoders as ce
    CE_OK = True
except Exception:
    CE_OK = False


# -----------------------------
# Streamlit settings
# -----------------------------
st.set_page_config(
    page_title="Patient ML Dashboard",
    page_icon="🫀",
    layout="wide"
)
st.title("🫀 Patient ML — EDA • Encodings • PCA • Baselines")

# -----------------------------
# Data loading
# -----------------------------
st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
default_path = "patient_dataset.csv"

@st.cache_data(show_spinner=False)
def load_data(file_like_or_path: str | io.BytesIO) -> pd.DataFrame:
    if isinstance(file_like_or_path, (str, os.PathLike)):
        return pd.read_csv(file_like_or_path)
    return pd.read_csv(file_like_or_path)

if uploaded:
    data = load_data(uploaded)
else:
    if os.path.exists(default_path):
        data = load_data(default_path)
    else:
        st.error("No data found. Please upload a CSV named `patient_dataset.csv` or use the uploader.")
        st.stop()

st.sidebar.success(f"Dataset loaded: {data.shape[0]} rows × {data.shape[1]} cols")

# Choose target
default_target = "heart_disease" if "heart_disease" in data.columns else data.columns[-1]
target_col = st.sidebar.selectbox("Target column", options=list(data.columns), index=list(data.columns).index(default_target))

# Select numeric / categorical
num_cols = list(data.select_dtypes(include=[np.number]).columns)
cat_cols = [c for c in data.columns if c not in num_cols and c != target_col]
num_cols = [c for c in num_cols if c != target_col]

with st.expander("Detected columns", expanded=False):
    st.write("**Numeric:**", num_cols)
    st.write("**Categorical:**", cat_cols)
    st.write("**Target:**", target_col)

# Drop rows with target missing
data = data.dropna(subset=[target_col]).copy()

X = data[num_cols + cat_cols].copy()
y = data[target_col].copy()

# -----------------------------
# Common preprocessors
# -----------------------------

preprocess_lr = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ],
    remainder="drop"
)

preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ],
    remainder="drop"
)

preprocess_ordinal_lr = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ],
    remainder="drop"
)

if CE_OK:
    preprocess_binary_lr = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("bin", ce.BinaryEncoder()),
            ]), cat_cols),
        ],
        remainder="drop"
    )
else:
    preprocess_binary_lr = None

# -----------------------------
# Utility functions
# -----------------------------
def eval_pipe(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv=5) -> tuple[float, float]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(yte, preds)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean() if proba is not None else np.nan
    return acc, auc

def choose_k_90(pca) -> int:
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cum, 0.90) + 1)

def plot_corr_circle(loadings, expl_ratio, pcs=(1,2), figsize=(5.8,5.8)):
    pcx, pcy = pcs[0]-1, pcs[1]-1
    plt.figure(figsize=figsize)
    ang = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(ang), np.sin(ang), color="lightgrey")
    for name, (x, y) in loadings[[f"PC{pcs[0]}", f"PC{pcs[1]}"]].iterrows():
        plt.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='tab:blue', ec='tab:blue', alpha=0.85, length_includes_head=True)
        plt.text(x*1.07, y*1.07, name, ha='center', va='center', fontsize=9)
    plt.axhline(0, color="grey", lw=0.7); plt.axvline(0, color="grey", lw=0.7)
    plt.xlim(-1.05, 1.05); plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f"PC{pcs[0]} ({expl_ratio[pcx]*100:.1f}%)")
    plt.ylabel(f"PC{pcs[1]} ({expl_ratio[pcy]*100:.1f}%)")
    plt.title("Correlation Circle")
    st.pyplot(plt.gcf())
    plt.close()

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["EDA", "Encoding Benchmarks", "PCA Analysis", "PC1–PC2 Model", "Baseline Models"])

# --- EDA tab ---
with tabs[0]:
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(20), width="stretch")

    # Missing values summary
    miss = data.isnull().mean().mul(100).round(2)
    miss_df = miss[miss > 0].sort_values(ascending=False).to_frame("Missing %")
    st.markdown("**Missing values**")
    if miss_df.empty:
        st.info("No missing values.")
    else:
        st.dataframe(miss_df, width="stretch")

    # Quick distributions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric distributions**")
        if len(num_cols) > 0:
            fig, axes = plt.subplots(len(num_cols), 1, figsize=(6.5, 3.5 * len(num_cols)))
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                sns.histplot(data[col].dropna(), bins=20, kde=True, ax=ax, color="tab:blue")
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numeric columns detected.")

    with col2:
        st.markdown("**Target balance**")
        fig, ax = plt.subplots(figsize=(5, 4))
        data[target_col].value_counts().plot(kind="bar", ax=ax)
        plt.title(f"Target: {target_col}")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Optional: Auto EDA Report (ydata_profiling)**")
    if YDATA_OK and st.button("Generate ydata_profiling report"):
        try:
            profile = ProfileReport(data, title="Rapport EDA - Patient Dataset", explorative=True)
            st.download_button("Download report (HTML)", data=profile.to_html(), file_name="rapport_eda.html")
        except Exception as e:
            st.error(f"Profiling failed: {e}")
    elif not YDATA_OK:
        st.caption("Install `ydata-profiling` to enable this button.")

# --- Encoding Benchmarks tab ---
with tabs[1]:
    st.subheader("Encoding: Label/Ordinal vs One-Hot vs Binary (Logistic Regression)")

    results = []
    models_enc = [
        ("Ordinal + LR", Pipeline([
            ("prep", preprocess_ordinal_lr),
            ("clf", LogisticRegression(max_iter=1000))
        ])),
        ("One-Hot + LR", Pipeline([
            ("prep", preprocess_lr),
            ("clf", LogisticRegression(max_iter=1000))
        ]))
    ]
    if CE_OK and preprocess_binary_lr is not None:
        models_enc.append((
            "BinaryEncoder + LR",
            Pipeline([
                ("prep", preprocess_binary_lr),
                ("clf", LogisticRegression(max_iter=1000))
            ])
        ))
    else:
        st.caption("Binary encoding not available — install `category-encoders` if you want this variant.")

    for name, pipe in models_enc:
        try:
            acc, auc = eval_pipe(pipe, X, y, cv=5)
            results.append((name, acc, auc))
        except Exception as e:
            results.append((name, np.nan, np.nan))
            st.error(f"{name} failed: {e}")

    res_df = pd.DataFrame(results, columns=["Encoding + Model", "Accuracy (test)", "ROC-AUC (CV=5)"])
    st.dataframe(res_df.style.format({"Accuracy (test)": "{:.3f}", "ROC-AUC (CV=5)": "{:.3f}"}), width="stretch")

# --- PCA Analysis tab ---
with tabs[2]:
    st.subheader("PCA: variance, loadings, correlation circle, 2D scatter")

    X_pca_base = data[num_cols].dropna().copy()
    y_pca = data.loc[X_pca_base.index, target_col]

    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X_pca_base)

    pca_full = PCA(n_components=None, random_state=42).fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k90 = int(np.searchsorted(cum, 0.90) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(cum, marker='o')
    ax.axhline(0.90, color='grey', ls='--', lw=1, label='90% threshold')
    ax.axvline(k90-1, color='tab:orange', ls='--', lw=1)
    ax.scatter([k90-1], [cum[k90-1]], color='tab:orange', zorder=3, label=f'k={k90}')
    ax.set_xlabel("Number of Components"); ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Explained Variance by Components"); ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig); plt.close(fig)

    pca_k = PCA(n_components=k90, random_state=42).fit(X_scaled)
    comps = pca_k.components_
    eigvals = pca_k.explained_variance_
    corr = comps.T * np.sqrt(eigvals)

    loadings = pd.DataFrame(corr, index=num_cols, columns=[f"PC{i+1}" for i in range(k90)])
    st.markdown("**Correlation of variables with PCs (loadings)**")
    st.dataframe(loadings.round(3), width="stretch")

    st.markdown("**Correlation circle (PC1 vs PC2)**")
    plot_corr_circle(loadings, pca_k.explained_variance_ratio_, pcs=(1,2))

    X_2D = pca_k.transform(X_scaled)[:, :2]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=X_2D[:,0], y=X_2D[:,1], hue=y_pca.astype(str), s=10, alpha=0.7, palette="Set1", ax=ax, edgecolor="none")
    ax.set_xlabel(f"PC1 ({pca_k.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_k.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Projection (PC1 vs PC2)")
    ax.legend(title=target_col, bbox_to_anchor=(1.02,1), loc="upper left")
    st.pyplot(fig); plt.close(fig)

    st.markdown("**Optional non-linear views (t-SNE / UMAP)**")
    c1, c2 = st.columns(2)
    with c1:
        if TSNE_OK and st.button("Run t-SNE"):
            try:
                X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
                fig, ax = plt.subplots(figsize=(6,5))
                sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_pca.astype(str), s=10, alpha=0.7, palette="Set1", ax=ax, edgecolor="none")
                ax.set_title("t-SNE Projection")
                ax.legend(title=target_col, bbox_to_anchor=(1.02,1), loc="upper left")
                st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"t-SNE failed: {e}")
    with c2:
        if UMAP_OK and st.button("Run UMAP"):
            try:
                X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(X_scaled)
                fig, ax = plt.subplots(figsize=(6,5))
                sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y_pca.astype(str), s=10, alpha=0.7, palette="coolwarm", ax=ax, edgecolor="none")
                ax.set_title("UMAP Projection")
                ax.legend(title=target_col, bbox_to_anchor=(1.02,1), loc="upper left")
                st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"UMAP failed: {e}")

# --- PC1–PC2 Model tab ---
with tabs[3]:
    st.subheader("Classifier on PC1 + PC2 (Logistic Regression)")

    X_pca_base = data[num_cols].dropna().copy()
    y_pca = data.loc[X_pca_base.index, target_col]
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X_pca_base)

    pca2 = PCA(n_components=2, random_state=42).fit(X_scaled)
    X_pc12 = pca2.transform(X_scaled)

    Xtr, Xte, ytr, yte = train_test_split(X_pc12, y_pca, test_size=0.2, random_state=42, stratify=y_pca)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:,1]

    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)

    st.write(f"**Accuracy (test)**: {acc:.3f}")
    st.write(f"**ROC-AUC (test)**: {auc:.3f}")
    st.code(classification_report(yte, pred), language="text")

# --- Baseline Models tab ---
with tabs[4]:
    st.subheader("Baselines")

    models = {
        "LogReg (One-Hot + Scale)": Pipeline([
            ("prep", preprocess_lr),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "RandomForest (Impute + Encode)": Pipeline([
            ("prep", preprocess_tree),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocess_tree),
            ("clf", HistGradientBoostingClassifier(random_state=42))
        ])
    }

    rows = []
    for name, pipe in models.items():
        try:
            acc, auc = eval_pipe(pipe, X, y, cv=5)
            rows.append((name, acc, auc))
        except Exception as e:
            rows.append((name, np.nan, np.nan))
            st.error(f"{name} failed: {e}")