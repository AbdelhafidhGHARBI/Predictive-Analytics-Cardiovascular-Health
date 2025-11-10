# src/train.py
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    # Sécurité: supprimer les doublons exacts
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def split_xy(df: pd.DataFrame, target: str = "heart_disease"):
    assert target in df.columns, f"Target '{target}' absente."
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def get_columns(X: pd.DataFrame):
    # Colonnes attendues (adapter au besoin si schéma change)
    num_cols = [
        "age","blood_pressure","cholesterol","max_heart_rate",
        "plasma_glucose","skin_thickness","insulin","bmi","diabetes_pedigree"
    ]
    num_cols = [c for c in num_cols if c in X.columns]

    cat_cols = [
        "gender","chest_pain_type","exercise_angina",
        "residence_type","smoking_status","hypertension"
    ]
    cat_cols = [c for c in cat_cols if c in X.columns]

    return num_cols, cat_cols

def build_pipelines(num_cols, cat_cols):
    # Pipeline pour modèle linéaire
    num_pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocess_lr = ColumnTransformer([
        ("num", num_pipe_lr, num_cols),
        ("cat", cat_pipe_lr, cat_cols)
    ])

    pipe_lr = Pipeline([
        ("prep", preprocess_lr),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # Pipeline pour RandomForest (robuste aux outliers)
    num_pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # ("robust", RobustScaler())  # optionnel
    ])
    cat_pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocess_rf = ColumnTransformer([
        ("num", num_pipe_rf, num_cols),
        ("cat", cat_pipe_rf, cat_cols)
    ])

    pipe_rf = Pipeline([
        ("prep", preprocess_rf),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return pipe_lr, pipe_rf

def evaluate_model(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_score = None
    auc = np.nan
    ap = np.nan

    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = pipe.decision_function(X_test)
        except Exception:
            pass

    if y_score is not None:
        auc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)

    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    if not np.isnan(auc):
        print("ROC AUC  :", round(auc, 3))
        print("PR AUC   :", round(ap, 3))
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return auc

def main():
    df = load_data("data/patient_dataset.csv")
    X, y = split_xy(df, target="heart_disease")
    num_cols, cat_cols = get_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe_lr, pipe_rf = build_pipelines(num_cols, cat_cols)

    # Entraînement
    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)

    # Évaluation Holdout
    auc_lr = evaluate_model("Logistic Regression", pipe_lr, X_test, y_test)
    auc_rf = evaluate_model("RandomForest", pipe_rf, X_test, y_test)

    # Cross-val sur l’ensemble
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc_lr = cross_val_score(pipe_lr, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_auc_rf = cross_val_score(pipe_rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print("\nCV ROC AUC (LR):", cv_auc_lr.mean().round(3), "±", cv_auc_lr.std().round(3))
    print("CV ROC AUC (RF):", cv_auc_rf.mean().round(3), "±", cv_auc_rf.std().round(3))

    # Sauvegarde du meilleur
    best_pipe = pipe_rf if (auc_rf or 0) >= (auc_lr or 0) else pipe_lr
    joblib.dump(best_pipe, "models/model_rf_pipeline.joblib")
    print("✅ Modèle sauvegardé: models/model_rf_pipeline.joblib")

if __name__ == "__main__":
    main()
