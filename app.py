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

# Rutas relativas
BASE_DIR = Path(__file__).parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_DIR = BASE_DIR / "data"

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

def load_models(models_dir: str) -> Dict[str, object]:
    loaded = {}
    for model_name, candidates in MODEL_FILENAMES.items():
        path = find_model_file(models_dir, candidates)
        if path is not None:
            loaded[model_name] = load_joblib_obj(str(path))
            loaded[f"{model_name}__path"] = str(path)
    return loaded

def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    schema = {}
    if df_ref is None:
        mins_maxs = {
            "Age": (60, 90), "BMI": (15.0, 40.0), "AlcoholConsumption": (0.0, 20.0),
            "PhysicalActivity": (0.0, 10.0), "DietQuality": (0.0, 10.0), "SleepQuality": (4.0, 10.0),
            "SystolicBP": (90, 180), "DiastolicBP": (60, 120), "CholesterolTotal": (150.0, 300.0),
            "CholesterolLDL": (50.0, 200.0), "CholesterolHDL": (20.0, 100.0),
            "CholesterolTriglycerides": (50.0, 400.0), "MMSE": (0.0, 30.0),
            "FunctionalAssessment": (0.0, 10.0), "ADL": (0.0, 10.0),
        }
        for c in EXPECTED_FEATURES:
            if c in CATEGORICAL_COLS:
                schema[c] = {"type": "categorical", "options": [0, 1] if c == "Gender" else [0, 1, 2, 3]}
            elif c in BINARY_COLS:
                schema[c] = {"type": "binary", "options": [0, 1]}
            elif c in INT_LIKE_COLS:
                schema[c] = {"type": "int", "min": mins_maxs.get(c, (0, 100))[0], "max": mins_maxs.get(c, (0, 100))[1]}
            else:
                schema[c] = {"type": "float", "min": mins_maxs.get(c, (0.0, 1.0))[0], "max": mins_maxs.get(c, (0.0, 1.0))[1]}
        return schema

    for c in EXPECTED_FEATURES:
        col = df_ref[c]
        if c in CATEGORICAL_COLS:
            schema[c] = {"type": "categorical", "options": sorted([int(x) for x in col.dropna().unique().tolist()])}
        elif c in BINARY_COLS:
            schema[c] = {"type": "binary", "options": [0, 1]}
        elif pd.api.types.is_integer_dtype(col):
            schema[c] = {"type": "int", "min": int(col.min()), "max": int(col.max())}
        else:
            schema[c] = {"type": "float", "min": float(col.min()), "max": float(col.max())}
    return schema

# ==========================================================
# UI
# ==========================================================

ref_path_default = str(DEFAULT_DATA_DIR / "alzheimer_dataset.xlsx")
models_dir_default = str(DEFAULT_MODELS_DIR)

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    models_dir = st.text_input("Carpeta de modelos", models_dir_default)
    reference_data_path = st.text_input("Dataset de referencia", ref_path_default)
    st.caption("Se usa para validar columnas, tipos y rangos.")

    st.markdown("### Modelos disponibles")
    st.write("- Regresión logística")
    st.write("- Random Forest")
    st.write("- Clustering")

reference_df = load_reference_dataset(reference_data_path)
schema = build_schema_from_reference(reference_df)
models = load_models(models_dir)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">🧠 Alzheimer ML Dashboard</h1>
        <p style="margin:0.35rem 0 0 0; font-size:1.02rem; opacity:0.95;">Carga CSV, captura datos manualmente, valida tipos/rangos y ejecuta predicciones con una interfaz clara y profesional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Modelos cargados", len([k for k in models.keys() if not k.endswith("__path")]))
with col_b:
    st.metric("Variables de entrada", len(EXPECTED_FEATURES))
with col_c:
    st.metric("Claves del esquema", len(schema))

if reference_df is None:
    st.warning("No se encontró el dataset de referencia. La validación usará rangos genéricos.")
else:
    st.success(f"Dataset de referencia cargado: {reference_df.shape[0]:,} filas × {reference_df.shape[1]:,} columnas")

missing_models = [m for m in ["Regresión logística", "Random Forest", "Clustering"] if m not in models]
if missing_models:
    st.warning(f"Faltan modelos por cargar: {missing_models}. Revisa la carpeta de modelos.")

# Estado de sesión
if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}

st.markdown("---")
st.markdown("### 📚 Instrucciones")
st.markdown("""
1. **Configura las rutas** en la barra lateral
2. **Carga datos** desde CSV o captura manual
3. **Ejecuta predicciones** con uno o varios modelos
4. **Visualiza resultados** y descarga predicciones
""")
