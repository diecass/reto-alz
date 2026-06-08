import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

st.set_page_config(
    page_title="Alzheimer ML App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #0ea5e9 100%);
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.20);
            margin-bottom: 1rem;
        }
        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
        }
        .small-note {color: #64748b; font-size: 0.92rem;}
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            font-size: 0.8rem;
            margin-right: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RANDOM_STATE = 42

# ==========================================================
# RUTAS PARA GITHUB / STREAMLIT CLOUD
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

# Cambia aquí si en tu repo los modelos están en otra carpeta
# Ejemplos:
#   "models"
#   "artifacts/models"
#   "files/models"
MODELS_SUBDIR = "models"
DATA_SUBDIR = "data"

DEFAULT_MODELS_DIR = BASE_DIR / MODELS_SUBDIR
DEFAULT_DATA_DIR = BASE_DIR / DATA_SUBDIR

EXPECTED_FEATURES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
    "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness",
]

OPTIONAL_TARGET = "Diagnosis"
IDENTIFIER_COLS = ["PatientID", "DoctorInCharge"]
CATEGORICAL_COLS = ["Gender", "Ethnicity"]
BINARY_COLS = [
    "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
    "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness",
]
INT_LIKE_COLS = ["Age", "EducationLevel", "SystolicBP", "DiastolicBP"] + BINARY_COLS + CATEGORICAL_COLS

MODEL_FILENAMES = {
    "Regresión logística": ["modelo_regresion_logistica_alzheimer.pkl"],
    "Random Forest - GridSearch (pocos datos)": ["modelo_random_forest_pocos_datos.pkl"],
    "Random Forest - RandomizedSearch (más datos)": ["modelo_random_forest_mas_datos.pkl"],
    "Clustering": ["cluster_kmeans_full_model.pkl"],
}

FEATURE_LABELS = {
    "Age": "Edad", "Gender": "Género (0/1)", "Ethnicity": "Etnia (0/1/2/3)",
    "EducationLevel": "Nivel educativo (0/1/2/3)", "BMI": "IMC",
    "Smoking": "Tabaquismo (0/1)", "AlcoholConsumption": "Consumo de alcohol",
    "PhysicalActivity": "Actividad física", "DietQuality": "Calidad de dieta",
    "SleepQuality": "Calidad de sueño", "FamilyHistoryAlzheimers": "Antecedente familiar (0/1)",
    "CardiovascularDisease": "Enfermedad cardiovascular (0/1)", "Diabetes": "Diabetes (0/1)",
    "Depression": "Depresión (0/1)", "HeadInjury": "Lesión en la cabeza (0/1)",
    "Hypertension": "Hipertensión (0/1)", "SystolicBP": "Presión sistólica",
    "DiastolicBP": "Presión diastólica", "CholesterolTotal": "Colesterol total",
    "CholesterolLDL": "Colesterol LDL", "CholesterolHDL": "Colesterol HDL",
    "CholesterolTriglycerides": "Triglicéridos", "MMSE": "MMSE",
    "FunctionalAssessment": "Evaluación funcional", "MemoryComplaints": "Quejas de memoria (0/1)",
    "BehavioralProblems": "Problemas de conducta (0/1)", "ADL": "Actividades de la vida diaria",
    "Confusion": "Confusión (0/1)", "Disorientation": "Desorientación (0/1)",
    "PersonalityChanges": "Cambios de personalidad (0/1)",
    "DifficultyCompletingTasks": "Dificultad para completar tareas (0/1)",
    "Forgetfulness": "Olvidos (0/1)",
}

# ==========================================================
# UTILIDADES
# ==========================================================

def resolve_existing_path(*paths: Path) -> Optional[Path]:
    """Devuelve la primera ruta existente."""
    for p in paths:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_reference_dataset(data_path: str) -> Optional[pd.DataFrame]:
    p = Path(data_path)
    if p.exists() and p.suffix.lower() in [".xlsx", ".xls", ".csv"]:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        return pd.read_excel(p)
    return None

def find_model_file(models_dir: str, candidates: List[str]) -> Optional[Path]:
    base = Path(models_dir)
    for fname in candidates:
        p = base / fname
        if p.exists():
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_joblib_obj(path: str):
    return joblib.load(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def load_models(models_dir: str) -> Dict[str, object]:
    loaded = {}
    for model_name, candidates in MODEL_FILENAMES.items():
        path = find_model_file(models_dir, candidates)
        if path is not None:
            loaded[model_name] = load_joblib_obj(str(path))
            loaded[f"{model_name}__path"] = str(path)
    return loaded
