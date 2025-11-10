# app.py
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============== CONFIG & THEME ==============
st.set_page_config(
    page_title="Cardio Risk – Demo",
    page_icon="🩺",
    layout="wide"
)

# CSS léger (couleurs, cartes, titres)
st.markdown("""
<style>
:root {
  --primary:#2563eb; /* bleu */
  --accent:#9333ea;  /* violet */
}
section.main > div {padding-top:1rem;}
.block-title {font-size:1.15rem; font-weight:600; margin-bottom:0.5rem;}
.card {border-radius:16px; padding:16px 18px; background:#ffffff; box-shadow:0 3px 20px rgba(0,0,0,0.06); border:1px solid rgba(0,0,0,0.05);}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:.9rem;}
.badge-green {background:#ecfdf5; color:#065f46;}
.badge-red {background:#fef2f2; color:#991b1b;}
.small {color:#6b7280; font-size:.9rem;}
hr{border:none; border-top:1px solid #e5e7eb; margin:12px 0 8px;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Predictive Analytics for Cardiovascular Health — Demo")

st.write("Cette démo illustre ton pipeline ML (prétraitement + modèle) et propose une visualisation PCA simple.")
st.write("— Choisis les paramètres du patient à gauche, puis regarde la probabilité prédite et la position du patient sur la PCA.")

# ============== HELPERS ==============
@st.cache_resource
def load_pipeline(path="model_rf_pipeline.joblib"):
    if not os.path.exists(path):
        return None, f"Le fichier `{path}` est introuvable dans le dossier courant."
    try:
        pipe = joblib.load(path)
        return pipe, None
    except Exception as e:
        return None, f"Erreur au chargement du modèle (`{path}`) : {e}"

@st.cache_resource
def load_dataset(path="patient_dataset.csv"):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def risk_label(p):
    if p >= 0.5:
        return "Risque élevé", "badge-red"
    else:
        return "Risque faible", "badge-green"

# ============== SIDEBAR – Inputs ==============
st.sidebar.header("🔧 Paramètres patient")

# Numériques (9 features principales)
age      = st.sidebar.slider("Age", 20, 90, 50)
bp       = st.sidebar.slider("Blood Pressure (mmHg)", 80, 220, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dL)", 100, 320, 200)
max_hr   = st.sidebar.slider("Max Heart Rate (bpm)", 60, 200, 145)
glucose  = st.sidebar.slider("Plasma Glucose (mg/dL)", 50, 300, 120)
skin     = st.sidebar.slider("Skin Thickness (mm)", 5, 80, 30)
insulin  = st.sidebar.slider("Insulin (μU/mL)", 0, 400, 100)
bmi      = st.sidebar.slider("BMI", 15.0, 50.0, 28.0, step=0.1)
dpf      = st.sidebar.slider("Diabetes Pedigree", 0.0, 3.0, 1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Catégorielles (cohérentes avec l'entraînement)")

gender          = st.sidebar.selectbox("Gender", ["Female","Male"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", [1,2,3,4])
exercise_angina = st.sidebar.selectbox("Exercise Angina", [0,1])
residence_type  = st.sidebar.selectbox("Residence Type", ["Urban","Rural"])
smoking_status  = st.sidebar.selectbox("Smoking Status", ["Non-Smoker","Smoker","Unknown"])
hypertension    = st.sidebar.selectbox("Hypertension", [0,1])

row = {
    "age": age,
    "blood_pressure": bp,
    "cholesterol": chol,
    "max_heart_rate": max_hr,
    "plasma_glucose": glucose,
    "skin_thickness": skin,
    "insulin": insulin,
    "bmi": bmi,
    "diabetes_pedigree": dpf,
    "gender": gender,
    "chest_pain_type": chest_pain_type,
    "exercise_angina": exercise_angina,
    "residence_type": residence_type,
    "smoking_status": smoking_status,
    "hypertension": hypertension
}
X_new = pd.DataFrame([row])

# ============== LOAD MODEL ==============
pipe, load_err = load_pipeline("model_rf_pipeline.joblib")
if load_err:
    st.error(f"❌ {load_err}")
    st.stop()

# ============== PREDICTION CARD ==============
with st.container():
    colA, colB = st.columns([1,2], gap="large")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="block-title">🎯 Prédiction</div>', unsafe_allow_html=True)

        proba = float(pipe.predict_proba(X_new)[0,1])
        pred = int(proba >= 0.5)

        label, badge_cls = risk_label(proba)
        st.markdown(f'<span class="badge {badge_cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(f"### Probabilité estimée : **{proba*100:.1f}%**")

        # barre de risque (0 → 1)
        st.progress(min(max(proba,0.0),1.0))

        # commentaire court
        if pred==1:
            st.write("**Interprétation rapide :** le modèle détecte un **profil à risque** sur la base des paramètres saisis.")
        else:
            st.write("**Interprétation rapide :** le modèle estime un **risque faible** pour les paramètres saisis.")

        st.markdown('<hr><div class="small">Note : ce score est indicatif et ne remplace pas un avis médical.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============== PCA PLOT (optionnel si dataset présent) ==============
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="block-title">📉 Visualisation PCA (population vs patient)</div>', unsafe_allow_html=True)

        df = load_dataset("patient_dataset.csv")
        if df is None:
            st.info("Ajoute `patient_dataset.csv` dans le dossier pour activer la PCA.")
        else:
            # on ne garde que les colonnes numériques utilisées
            num_cols = ["age","blood_pressure","cholesterol","max_heart_rate",
                        "plasma_glucose","skin_thickness","insulin","bmi","diabetes_pedigree"]
            num_cols = [c for c in num_cols if c in df.columns]

            # imputation + standardisation (fit sur population)
            imputer = SimpleImputer(strategy="median")
            scaler  = StandardScaler()

            Z = imputer.fit_transform(df[num_cols])
            Z = scaler.fit_transform(Z)

            # PCA 2D
            pca = PCA(n_components=2, random_state=42)
            Z2 = pca.fit_transform(Z)

            # transformer le patient
            x_patient = imputer.transform(X_new[num_cols])
            x_patient = scaler.transform(x_patient)
            p_patient = pca.transform(x_patient)[0]

            # scatter
            fig = plt.figure(figsize=(6.8,5.0))
            plt.scatter(Z2[:,0], Z2[:,1], s=10, alpha=0.25)
            plt.scatter(p_patient[0], p_patient[1], s=140, marker="X")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.title("Projection PCA — Position du patient (X)")
            st.pyplot(fig, clear_figure=True)

            st.markdown('<div class="small">La PCA est une **visualisation** (pas utilisée par le pipeline). Elle montre la position relative du patient vs. la variabilité globale.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============== EXPLAINABILITY LÉGÈRE (importances RF) ==============
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🧠 Indices d’importance (RandomForest)</div>', unsafe_allow_html=True)
    try:
        # on récupère la liste des features après prétraitement
        # ATTENTION : pour ColumnTransformer + OrdinalEncoder, la taille change;
        # ici on montre une importance agrégée "numérique" de manière simplifiée.
        clf = pipe.named_steps.get("clf", None)
        prep = pipe.named_steps.get("prep", None)

        if hasattr(clf, "feature_importances_") and prep is not None:
            # pour une lecture simple, on affiche au moins l’importance des 9 numériques (si présentes)
            raw_cols = list(X_new.columns)
            # importance globale (après encoding, dimensions > nb features brutes)
            importances = clf.feature_importances_
            # on ne peut pas remapper précisément toutes les colonnes encodées ici sans récupérer get_feature_names_out
            # on donne une info de niveau projet : score moyen par bloc (numérique vs catégoriel)
            num_idx = [i for i,c in enumerate(raw_cols) if c in ["age","blood_pressure","cholesterol","max_heart_rate",
                                                                 "plasma_glucose","skin_thickness","insulin","bmi","diabetes_pedigree"]]
            cat_idx = [i for i,c in enumerate(raw_cols) if c not in ["age","blood_pressure","cholesterol","max_heart_rate",
                                                                     "plasma_glucose","skin_thickness","insulin","bmi","diabetes_pedigree"]]
            st.write("**Note** : l’importance exacte par variable encodée nécessite `get_feature_names_out` du `ColumnTransformer`.")
            st.write("Ici, on affiche un rappel des features clés utilisées par le modèle :")
            st.code(", ".join(raw_cols), language="text")
        else:
            st.info("Les importances ne sont pas disponibles pour ce classifieur ou ce pipeline.")
    except Exception as e:
        st.warning(f"Impossible d'afficher les importances : {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============== FOOTER ==============
st.caption("© Demo académique — ce modèle n’est pas un dispositif médical.")
