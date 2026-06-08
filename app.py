import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

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
            margin-bottom: 0.25rem;
        }

        /* Botones con color más suave y accesible */
        div.stButton > button,
        div.stDownloadButton > button {
            background-color: #2563eb !important;
            color: white !important;
            border: 1px solid #1d4ed8 !important;
            border-radius: 12px !important;
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background-color: #1d4ed8 !important;
            color: white !important;
            border-color: #1e40af !important;
        }
        div.stButton > button:focus,
        div.stDownloadButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(37, 99, 235, 0.25) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
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
    "Regresión logística": [
        "modelo_regresion_logistica_alzheimer.pkl",
    ],
    "Random Forest - GridSearch (pocos datos)": [
        "modelo_random_forest_pocos_datos.pkl",
    ],
    "Random Forest - RandomizedSearch (más datos)": [
        "modelo_random_forest_mas_datos.pkl",
    ],
    "Clustering": [
        "cluster_kmeans_full_model.pkl",
    ],
}

FEATURE_LABELS = {
    "Age": "Edad",
    "Gender": "Género (0/1)",
    "Ethnicity": "Etnia (0/1/2/3)",
    "EducationLevel": "Nivel educativo (0/1/2/3)",
    "BMI": "IMC",
    "Smoking": "Tabaquismo (0/1)",
    "AlcoholConsumption": "Consumo de alcohol",
    "PhysicalActivity": "Actividad física",
    "DietQuality": "Calidad de dieta",
    "SleepQuality": "Calidad de sueño",
    "FamilyHistoryAlzheimers": "Antecedente familiar (0/1)",
    "CardiovascularDisease": "Enfermedad cardiovascular (0/1)",
    "Diabetes": "Diabetes (0/1)",
    "Depression": "Depresión (0/1)",
    "HeadInjury": "Lesión en la cabeza (0/1)",
    "Hypertension": "Hipertensión (0/1)",
    "SystolicBP": "Presión sistólica",
    "DiastolicBP": "Presión diastólica",
    "CholesterolTotal": "Colesterol total",
    "CholesterolLDL": "Colesterol LDL",
    "CholesterolHDL": "Colesterol HDL",
    "CholesterolTriglycerides": "Triglicéridos",
    "MMSE": "MMSE",
    "FunctionalAssessment": "Evaluación funcional",
    "MemoryComplaints": "Quejas de memoria (0/1)",
    "BehavioralProblems": "Problemas de conducta (0/1)",
    "ADL": "Actividades de la vida diaria",
    "Confusion": "Confusión (0/1)",
    "Disorientation": "Desorientación (0/1)",
    "PersonalityChanges": "Cambios de personalidad (0/1)",
    "DifficultyCompletingTasks": "Dificultad para completar tareas (0/1)",
    "Forgetfulness": "Olvidos (0/1)",
}

# ==========================================================
# CARGA Y VALIDACIÓN
# ==========================================================

@st.cache_data(show_spinner=False)
def load_reference_dataset(data_path: str) -> Optional[pd.DataFrame]:
    p = Path(data_path)
    if not p.exists() or p.is_dir():
        return None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
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
    loaded: Dict[str, object] = {}
    for model_name, candidates in MODEL_FILENAMES.items():
        path = find_model_file(models_dir, candidates)
        if path is not None:
            loaded[model_name] = load_joblib_obj(str(path))
            loaded[f"{model_name}__path"] = str(path)
    return loaded

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    schema: Dict[str, Dict] = {}

    if df_ref is None:
        mins_maxs = {
            "Age": (60, 90), "BMI": (15.0, 40.0), "AlcoholConsumption": (0.0, 20.0),
            "PhysicalActivity": (0.0, 10.0), "DietQuality": (0.0, 10.0), "SleepQuality": (4.0, 10.0),
            "SystolicBP": (90, 180), "DiastolicBP": (60, 120), "CholesterolTotal": (150.0, 300.0),
            "CholesterolLDL": (50.0, 200.0), "CholesterolHDL": (20.0, 100.0), "CholesterolTriglycerides": (50.0, 400.0),
            "MMSE": (0.0, 30.0), "FunctionalAssessment": (0.0, 10.0), "ADL": (0.0, 10.0),
        }
        for c in EXPECTED_FEATURES:
            if c in CATEGORICAL_COLS:
                schema[c] = {"type": "categorical", "options": [0, 1] if c == "Gender" else [0, 1, 2, 3]}
            elif c in BINARY_COLS:
                schema[c] = {"type": "binary", "options": [0, 1]}
            elif c in INT_LIKE_COLS:
                low, high = mins_maxs.get(c, (0, 100))
                schema[c] = {"type": "int", "min": low, "max": high}
            else:
                low, high = mins_maxs.get(c, (0.0, 1.0))
                schema[c] = {"type": "float", "min": low, "max": high}
        return schema

    for c in EXPECTED_FEATURES:
        if c not in df_ref.columns:
            continue
        col = pd.to_numeric(df_ref[c], errors="coerce") if c not in ["Gender", "Ethnicity"] else pd.to_numeric(df_ref[c], errors="coerce")
        if c in CATEGORICAL_COLS:
            vals = sorted([int(x) for x in col.dropna().astype(int).unique().tolist()])
            schema[c] = {"type": "categorical", "options": vals if len(vals) > 0 else ([0, 1] if c == "Gender" else [0, 1, 2, 3])}
        elif c in BINARY_COLS:
            schema[c] = {"type": "binary", "options": [0, 1]}
        elif pd.api.types.is_integer_dtype(df_ref[c]):
            schema[c] = {"type": "int", "min": int(pd.to_numeric(col, errors="coerce").min()), "max": int(pd.to_numeric(col, errors="coerce").max())}
        else:
            schema[c] = {"type": "float", "min": float(pd.to_numeric(col, errors="coerce").min()), "max": float(pd.to_numeric(col, errors="coerce").max())}

    # Asegurar que todas las columnas esperadas estén presentes
    fallback = build_schema_from_reference(None)
    for c in EXPECTED_FEATURES:
        if c not in schema:
            schema[c] = fallback[c]
    return schema

def validate_and_prepare_csv(df_in: pd.DataFrame, schema: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = normalize_columns(df_in)
    issues: List[str] = []

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_FEATURES + IDENTIFIER_COLS + [OPTIONAL_TARGET]]

    if missing:
        issues.append(f"Faltan columnas requeridas: {missing}")
    if extra:
        issues.append(f"Columnas extra que se ignorarán: {extra}")

    out = pd.DataFrame(index=df.index)

    for c in EXPECTED_FEATURES:
        if c not in df.columns:
            out[c] = np.nan
            continue

        s = df[c].copy()
        spec = schema[c]

        if spec["type"] in {"float", "int"}:
            s = coerce_numeric(s)
            if spec["type"] == "int":
                bad = s.dropna()[~np.isclose(s.dropna() % 1, 0)]
                if len(bad) > 0:
                    issues.append(f"{c}: hay valores no enteros en filas {bad.index.tolist()[:10]}")
                s = s.round(0)
            low, high = spec["min"], spec["max"]
            bad_range = s.dropna()[(s.dropna() < low) | (s.dropna() > high)]
            if len(bad_range) > 0:
                issues.append(f"{c}: valores fuera de rango [{low}, {high}] en filas {bad_range.index.tolist()[:10]}")
            out[c] = s
        else:
            s = coerce_numeric(s).round(0)
            allowed = set(spec["options"])
            bad_cat = s.dropna()[~s.dropna().isin(list(allowed))]
            if len(bad_cat) > 0:
                issues.append(f"{c}: valores no permitidos {sorted(set(bad_cat.tolist()))[:10]}")
            out[c] = s

    if OPTIONAL_TARGET in df.columns:
        out[OPTIONAL_TARGET] = coerce_numeric(df[OPTIONAL_TARGET]).round(0)

    validity = pd.DataFrame(index=out.index)
    validity["row_valid"] = True
    for c in EXPECTED_FEATURES:
        spec = schema[c]
        validity[f"{c}_valid"] = out[c].notna()
        if spec["type"] in {"float", "int"}:
            validity[f"{c}_valid"] &= (out[c].astype(float) >= spec["min"]) & (out[c].astype(float) <= spec["max"])
        else:
            validity[f"{c}_valid"] &= out[c].isin(spec["options"])
    validity["row_valid"] = validity[[f"{c}_valid" for c in EXPECTED_FEATURES]].all(axis=1)

    return out, validity, issues

def get_defaults_from_reference(reference_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    if reference_df is None:
        return defaults

    for c in EXPECTED_FEATURES:
        if c not in reference_df.columns:
            continue
        series = pd.to_numeric(reference_df[c], errors="coerce")
        if c in CATEGORICAL_COLS or c in BINARY_COLS or pd.api.types.is_integer_dtype(reference_df[c]):
            mode = series.dropna().mode()
            if len(mode) > 0:
                defaults[c] = int(mode.iloc[0])
        else:
            med = series.dropna().median()
            if pd.notna(med):
                defaults[c] = float(med)
    return defaults

def to_model_input(df_prepared: pd.DataFrame) -> pd.DataFrame:
    x = df_prepared[EXPECTED_FEATURES].copy()
    for c in INT_LIKE_COLS:
        if c in x.columns:
            x[c] = x[c].round(0).astype("Int64")
    for c in x.columns:
        if c not in INT_LIKE_COLS:
            x[c] = x[c].astype(float)
    return x

# ==========================================================
# MÉTRICAS Y PREDICCIÓN
# ==========================================================

def predict_supervised(model, x: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = model.predict(x)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)[:, 1]
        except Exception:
            probs = None
    return preds, probs

def predict_cluster(artifact, x: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    if isinstance(artifact, dict):
        pre = artifact.get("preprocessor")
        cluster_model = artifact.get("cluster_model")
        if pre is not None and cluster_model is not None:
            x_t = pre.transform(x)
            labels = cluster_model.predict(x_t) if hasattr(cluster_model, "predict") else cluster_model.fit_predict(x_t)
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(x_t)
            viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
            viz["Cluster"] = labels
            return labels, viz

    if hasattr(artifact, "predict"):
        labels = artifact.predict(x)
        try:
            if hasattr(artifact, "named_steps") and "preprocess" in artifact.named_steps:
                x_t = artifact.named_steps["preprocess"].transform(x)
            else:
                x_t = x.copy()
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(x_t)
            viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
            viz["Cluster"] = labels
            return labels, viz
        except Exception:
            return labels, pd.DataFrame({"Cluster": labels})

    raise ValueError("No se pudo interpretar el artefacto de clustering.")

def compute_supervised_metrics(y_true, y_pred, y_prob=None) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": np.nan,
        "cm": cm,
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc"] = np.nan
    return metrics

def plot_confusion_matrix(cm, title: str):
    tn, fp, fn, tp = cm.ravel()
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicho 0", "Predicho 1"],
            y=["Real 0", "Real 1"],
            text=z,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Valor real",
        height=450,
    )
    return fig

def plot_roc(y_true, y_prob, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Modelo"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="FPR", yaxis_title="TPR", height=450)
    return fig

def get_rf_importance(model) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        out = pd.DataFrame({"feature": feature_names, "importance": importances})
        return out.sort_values("importance", ascending=False)
    except Exception:
        return None

def get_lr_coefficients(model) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        coef = model.named_steps["model"].coef_[0]
        out = pd.DataFrame({"feature": feature_names, "coef": coef})
        out["abs_coef"] = out["coef"].abs()
        return out.sort_values("abs_coef", ascending=False)
    except Exception:
        return None

def get_input_data(source_choice: str) -> Optional[pd.DataFrame]:
    csv_df = st.session_state.get("validated_csv")
    man_df = st.session_state.get("manual_df")
    parts = []

    if source_choice in ["CSV validado", "Ambos"] and csv_df is not None and len(csv_df) > 0:
        parts.append(csv_df[EXPECTED_FEATURES].copy())

    if source_choice in ["Registros manuales", "Ambos"] and man_df is not None and len(man_df) > 0:
        parts.append(man_df[EXPECTED_FEATURES].copy())

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)

def display_metric_cards(metrics: Dict[str, object]):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Sensibilidad", f"{metrics['recall']:.3f}")
    c4.metric("Especificidad", f"{metrics['specificity']:.3f}")
    c5.metric("F1", f"{metrics['f1']:.3f}")

# ==========================================================
# ESTADO DE SESIÓN
# ==========================================================

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "manual_df" not in st.session_state:
    st.session_state["manual_df"] = pd.DataFrame(columns=EXPECTED_FEATURES)
if "validated_csv" not in st.session_state:
    st.session_state["validated_csv"] = None
if "csv_issues" not in st.session_state:
    st.session_state["csv_issues"] = []
if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}
if "cluster_results" not in st.session_state:
    st.session_state["cluster_results"] = {}

# ==========================================================
# CARGA DE RECURSOS
# ==========================================================

ref_path_default = str(DEFAULT_DATA_DIR / "alzheimer_dataset.xlsx")
models_dir_default = str(DEFAULT_MODELS_DIR)

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    models_dir = st.text_input("Carpeta de modelos", models_dir_default)
    reference_data_path = st.text_input("Dataset de referencia", ref_path_default)
    st.caption("Se usa para validar columnas, tipos y rangos.")

reference_df = load_reference_dataset(reference_data_path)
schema = build_schema_from_reference(reference_df)
models = load_models(models_dir)

st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">🧠 Alzheimer ML Dashboard</h2>
        <p style="margin:0.35rem 0 0 0;">
            Carga CSV, captura datos manuales, ejecuta modelos supervisados y clustering, y revisa hallazgos en una sola app.
        </p>
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

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Cargar CSV",
    "✍️ Captura manual",
    "🤖 Regresión",
    "👥 Grupos",
    "📊 Hallazgos",
])

# ==========================================================
# TAB 1: CSV
# ==========================================================

with tab1:
    st.subheader("Carga de archivo CSV")
    uploaded = st.file_uploader("Sube un archivo CSV con las variables del modelo", type=["csv"])

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df_raw

            prepared, validity, issues = validate_and_prepare_csv(df_raw, schema)
            st.session_state["validated_csv"] = prepared
            st.session_state["csv_issues"] = issues

            st.success(f"Archivo cargado correctamente: {df_raw.shape[0]:,} filas × {df_raw.shape[1]:,} columnas")
            st.dataframe(df_raw.head(20), use_container_width=True)

            if issues:
                st.warning("Se detectaron observaciones de validación.")
                for i in issues:
                    st.write(f"- {i}")
            else:
                st.info("No se detectaron problemas de columnas, tipos ni rangos.")

            valid_rate = validity["row_valid"].mean() * 100
            st.metric("Filas válidas", f"{valid_rate:.1f}%")

            st.markdown("#### Resumen de columnas")
            summary = pd.DataFrame({
                "columna": EXPECTED_FEATURES,
                "tipo": [schema[c]["type"] for c in EXPECTED_FEATURES],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

# ==========================================================
# TAB 2: CAPTURA MANUAL
# ==========================================================

with tab2:
    st.subheader("Captura manual de un registro")
    st.write("Completa el formulario y agrega el registro a la cola de predicción.")

    defaults = get_defaults_from_reference(reference_df)

    with st.form("manual_form", clear_on_submit=False):
        cols = st.columns(3)
        record = {}

        def input_widget(col, feature: str, spec: Dict[str, object]):
            label = FEATURE_LABELS.get(feature, feature)
            default_value = defaults.get(feature)

            if spec["type"] in {"binary", "categorical"}:
                opts = spec["options"]
                index = opts.index(default_value) if default_value in opts else 0
                return col.selectbox(label, opts, index=index)

            if spec["type"] == "int":
                mn, mx = int(spec["min"]), int(spec["max"])
                value = int(default_value) if default_value is not None else mn
                return col.number_input(label, min_value=mn, max_value=mx, value=value, step=1)

            mn, mx = float(spec["min"]), float(spec["max"])
            value = float(default_value) if default_value is not None else mn
            step = (mx - mn) / 100.0 if mx > mn else 0.1
            return col.number_input(label, min_value=mn, max_value=mx, value=value, step=step)

        for idx, feature in enumerate(EXPECTED_FEATURES):
            c = cols[idx % 3]
            record[feature] = input_widget(c, feature, schema[feature])

        submitted = st.form_submit_button("Agregar registro", use_container_width=True)

    if submitted:
        new_row = pd.DataFrame([record])
        st.session_state["manual_df"] = pd.concat([st.session_state["manual_df"], new_row], ignore_index=True)
        st.success("Registro agregado correctamente.")

    st.markdown("#### Registros manuales almacenados")
    if len(st.session_state["manual_df"]) > 0:
        st.dataframe(st.session_state["manual_df"], use_container_width=True)
        if st.button("Limpiar registros manuales"):
            st.session_state["manual_df"] = pd.DataFrame(columns=EXPECTED_FEATURES)
            st.rerun()
    else:
        st.info("Aún no has agregado registros manuales.")

# ==========================================================
# TAB 3: REGRESIÓN
# ==========================================================

with tab3:
    st.subheader("Modelos supervisados")
    st.markdown(
        """
        <div class="card">
            <span class="badge">Predicción clínica</span>
            <span class="badge">Curva ROC</span>
            <span class="badge">Matriz de confusión</span>
            <span class="badge">CSV o manual</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Esta pestaña ejecuta únicamente los tres modelos de regresión y guarda sus resultados
                para mostrarlos en Hallazgos. Si el CSV trae la columna <b>Diagnosis</b>, se calculan
                las métricas reales; si no, la salida queda como predicción descriptiva.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Fuente de datos",
        ["CSV validado", "Registros manuales", "Ambos"],
        horizontal=True,
    )

    sensitivity_threshold = st.slider(
        "Umbral para clasificar 'Tiene diagnóstico'",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.01,
        help="Mientras más bajo sea el umbral, más fácil será marcar un caso como positivo. Esto aumenta la sensibilidad y puede reducir la precisión.",
    )

    available_supervised = [
        m for m in [
            "Regresión logística",
            "Random Forest - GridSearch (pocos datos)",
            "Random Forest - RandomizedSearch (más datos)",
        ] if m in models
    ]

    selected_models = st.multiselect(
        "Selecciona los modelos a ejecutar",
        available_supervised,
        default=available_supervised,
    )

    run_btn = st.button("Ejecutar predicción", use_container_width=True)

    if run_btn:
        data_in = get_input_data(source)

        if data_in is None or len(data_in) == 0:
            st.error("No hay datos para predecir. Sube un CSV válido o agrega registros manuales.")
        elif len(selected_models) == 0:
            st.error("Selecciona al menos un modelo.")
        else:
            x_in = to_model_input(data_in)
            st.success(f"Datos listos para predicción: {x_in.shape[0]:,} filas × {x_in.shape[1]:,} variables")
            st.dataframe(x_in.head(20), use_container_width=True)

            uploaded_df = st.session_state.get("uploaded_df")
            validated_csv = st.session_state.get("validated_csv")

            y_true = None
            if source == "CSV validado" and uploaded_df is not None and validated_csv is not None:
                if OPTIONAL_TARGET in uploaded_df.columns and len(uploaded_df) == len(validated_csv):
                    y_true = pd.to_numeric(uploaded_df[OPTIONAL_TARGET], errors="coerce").fillna(0).astype(int).values

            for model_name in selected_models:
                st.markdown(f"### {model_name}")
                model_obj = models[model_name]

                preds, probs = predict_supervised(model_obj, x_in)
                clinical_pred = np.where(probs >= sensitivity_threshold, 1, 0) if probs is not None else preds.copy()

                out = x_in.copy()
                out[f"Pred_{model_name}"] = preds
                if probs is not None:
                    out[f"Prob_{model_name}"] = probs
                out[f"Diagnóstico_clínico_{model_name}"] = np.where(
                    clinical_pred == 1,
                    "Tiene diagnóstico",
                    "No tiene diagnóstico"
                )

                st.session_state["model_results"][model_name] = {
                    "kind": "supervised",
                    "preds": preds,
                    "probs": probs,
                    "clinical_pred": clinical_pred,
                    "threshold": sensitivity_threshold,
                    "y_true": y_true,
                    "source": source,
                }

                c1, c2, c3 = st.columns(3)
                c1.metric("Tiene diagnóstico", int(np.sum(clinical_pred == 1)))
                c2.metric("No tiene diagnóstico", int(np.sum(clinical_pred == 0)))
                c3.metric("Probabilidad media", f"{np.mean(probs):.3f}" if probs is not None else "N/A")

                st.dataframe(out.head(50), use_container_width=True)

                summary = pd.DataFrame({
                    "Resultado clínico": ["Tiene diagnóstico", "No tiene diagnóstico"],
                    "Cantidad": [int(np.sum(clinical_pred == 1)), int(np.sum(clinical_pred == 0))],
                })
                fig_summary = px.bar(summary, x="Resultado clínico", y="Cantidad", title=f"Resumen clínico - {model_name}")
                st.plotly_chart(fig_summary, use_container_width=True)

                if y_true is not None and len(y_true) == len(clinical_pred):
                    metrics = compute_supervised_metrics(y_true, clinical_pred, probs)
                    display_metric_cards(metrics)

                    st.plotly_chart(plot_confusion_matrix(metrics["cm"], f"Matriz de confusión - {model_name}"), use_container_width=True)

                    if probs is not None and len(np.unique(y_true)) > 1:
                        st.plotly_chart(plot_roc(y_true, probs, f"Curva ROC - {model_name}"), use_container_width=True)

                    st.write(
                        f"El modelo **{model_name}** prioriza la detección de casos positivos cuando el umbral es {sensitivity_threshold:.2f}. "
                        f"Con este criterio, los resultados se guardaron para la tab de Hallazgos."
                    )
                else:
                    st.info(
                        "No existe columna real de diagnóstico para este conjunto de datos, así que solo se muestra el resultado predictivo y la distribución de casos."
                    )

                if "Random Forest" in model_name:
                    imp = get_rf_importance(model_obj)
                    if imp is not None:
                        fig_imp = px.bar(
                            imp.head(15).iloc[::-1],
                            x="importance",
                            y="feature",
                            orientation="h",
                            title=f"Importancia de variables - {model_name}",
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                if model_name == "Regresión logística":
                    coef = get_lr_coefficients(model_obj)
                    if coef is not None:
                        top = coef.sort_values("abs_coef", ascending=False).head(15).copy()
                        top["direction"] = np.where(top["coef"] >= 0, "Positivo", "Negativo")
                        fig_coef = px.bar(
                            top.iloc[::-1],
                            x="coef",
                            y="feature",
                            orientation="h",
                            color="direction",
                            title="Coeficientes más relevantes - Regresión logística",
                        )
                        st.plotly_chart(fig_coef, use_container_width=True)

                st.markdown("---")

            st.markdown("#### Descarga de resultados")
            merged = pd.concat(
                [x_in] + [
                    pd.DataFrame({
                        f"Pred_{name}": st.session_state["model_results"][name]["preds"],
                        **(
                            {f"Prob_{name}": st.session_state["model_results"][name]["probs"]}
                            if st.session_state["model_results"][name]["probs"] is not None else {}
                        ),
                        f"Diagnóstico_clínico_{name}": np.where(
                            st.session_state["model_results"][name]["clinical_pred"] == 1,
                            "Tiene diagnóstico",
                            "No tiene diagnóstico",
                        ),
                    })
                    for name in selected_models
                ],
                axis=1,
            )

            st.download_button(
                "Descargar resultados como CSV",
                data=merged.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_alzheimer.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ==========================================================
# TAB 4: GRUPOS
# ==========================================================

with tab4:
    st.subheader("Grupos (Clustering)")
    st.markdown(
        """
        <div class="card">
            <span class="badge">Segmentación</span>
            <span class="badge">PCA visual</span>
            <span class="badge">Solo cluster</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Esta pestaña quedó solo para agrupar pacientes por similitud. No usa los datos manuales
                ni mezclas, porque el análisis de grupos se interpreta mejor sobre el CSV validado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "Clustering" not in models:
        st.error("No se encontró el modelo de clustering en la carpeta configurada.")
    elif st.session_state.get("validated_csv") is None:
        st.warning("Primero carga y valida un CSV para poder ejecutar el clustering.")
    else:
        st.info("El clustering se ejecuta únicamente con el CSV validado.")

        run_cluster = st.button("Ejecutar clustering", use_container_width=True)

        if run_cluster:
            x_in = to_model_input(st.session_state["validated_csv"])
            labels, viz = predict_cluster(models["Clustering"], x_in)

            out = x_in.copy()
            out["Cluster"] = labels
            st.session_state["cluster_results"]["Clustering"] = {
                "kind": "cluster",
                "labels": labels,
                "viz": viz,
                "data": out,
            }

            c1, c2, c3 = st.columns(3)
            c1.metric("Número de clusters", int(pd.Series(labels).nunique()))
            c2.metric("Cluster más frecuente", int(pd.Series(labels).mode().iloc[0]))
            c3.metric("Registros analizados", int(len(labels)))

            st.dataframe(out.head(50), use_container_width=True)

            counts = pd.Series(labels).value_counts().sort_index().reset_index()
            counts.columns = ["Cluster", "Cantidad"]
            fig_counts = px.bar(counts, x="Cluster", y="Cantidad", title="Tamaño de clusters")
            st.plotly_chart(fig_counts, use_container_width=True)

            if {"PC1", "PC2"}.issubset(viz.columns):
                fig_scatter = px.scatter(
                    viz,
                    x="PC1",
                    y="PC2",
                    color=viz["Cluster"].astype(str),
                    title="Visualización PCA de los clusters",
                    opacity=0.85,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.success("El clustering quedó guardado para mostrarse también en Hallazgos.")

# ==========================================================
# TAB 5: HALLAZGOS
# ==========================================================

with tab5:
    st.subheader("Hallazgos más relevantes")
    st.markdown(
        """
        <div class="card">
            <span class="badge">Comparación entre modelos</span>
            <span class="badge">Resumen clínico</span>
            <span class="badge">Clustering</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Aquí se concentran los resúmenes: la comparación de los tres modelos supervisados,
                la tabla de métricas y la gráfica del clustering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------
    # Resumen supervisado
    # ---------------------------
    supervised_entries = []
    for model_name, info in st.session_state.get("model_results", {}).items():
        if info.get("kind") != "supervised":
            continue

        y_pred = info.get("clinical_pred", info.get("preds"))
        y_prob = info.get("probs")
        y_true = info.get("y_true")

        if y_true is not None and len(y_true) == len(y_pred):
            metrics = compute_supervised_metrics(y_true, y_pred, y_prob)
            supervised_entries.append({
                "Modelo": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Sensibilidad": metrics["recall"],
                "Especificidad": metrics["specificity"],
                "F1": metrics["f1"],
                "AUC": metrics["auc"],
            })
        else:
            supervised_entries.append({
                "Modelo": model_name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Sensibilidad": np.nan,
                "Especificidad": np.nan,
                "F1": np.nan,
                "AUC": np.nan,
            })

    if supervised_entries:
        st.markdown("### Comparación entre los tres modelos")
        df_supervised = pd.DataFrame(supervised_entries)

        sort_cols = [c for c in ["Sensibilidad", "F1", "Especificidad", "Accuracy"] if c in df_supervised.columns]
        if df_supervised["Sensibilidad"].notna().any() and len(sort_cols) > 0:
            df_supervised = df_supervised.sort_values(by=sort_cols, ascending=False)

        st.dataframe(
            df_supervised.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Sensibilidad": "{:.3f}",
                "Especificidad": "{:.3f}",
                "F1": "{:.3f}",
                "AUC": "{:.3f}",
            }),
            use_container_width=True,
        )

        metric_cols = [c for c in ["Accuracy", "Precision", "Sensibilidad", "Especificidad", "F1", "AUC"] if c in df_supervised.columns]
        if metric_cols:
            plot_df = df_supervised.melt(id_vars="Modelo", value_vars=metric_cols, var_name="Métrica", value_name="Valor")
            fig_metrics = px.bar(
                plot_df,
                x="Modelo",
                y="Valor",
                color="Métrica",
                barmode="group",
                title="Comparación de desempeño entre modelos supervisados",
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        if df_supervised["Sensibilidad"].notna().any():
            best_row = df_supervised.sort_values(
                by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
                ascending=False
            ).iloc[0]
            st.success(
                f"Modelo seleccionado por sensibilidad: **{best_row['Modelo']}** "
                f"(Sensibilidad={best_row['Sensibilidad']:.3f}, F1={best_row['F1']:.3f})."
            )
        else:
            st.info("No hay métricas reales disponibles todavía; ejecuta los modelos con un CSV que incluya la columna Diagnosis.")

    else:
        st.info("No se han ejecutado modelos supervisados todavía.")

    # ---------------------------
    # Resumen clustering
    # ---------------------------
    cluster_info = st.session_state.get("cluster_results", {}).get("Clustering")

    if cluster_info is not None:
        st.markdown("### Clustering")
        labels = cluster_info.get("labels")
        out = cluster_info.get("data")
        counts = pd.Series(labels).value_counts().sort_index().reset_index()
        counts.columns = ["Cluster", "Cantidad"]

        fig_pie = px.pie(
            counts,
            names="Cluster",
            values="Cantidad",
            title="Distribución de clusters",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.dataframe(out.head(50), use_container_width=True)
    else:
        st.info("Todavía no se ha ejecutado el clustering.")

    st.markdown("### Conclusión ejecutiva")
    if supervised_entries and any(pd.notna(x) for x in df_supervised["Sensibilidad"].tolist()):
        best_exec = df_supervised.sort_values(
            by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
            ascending=False
        ).iloc[0]
        st.success(
            f"En términos clínicos, el modelo más conveniente es **{best_exec['Modelo']}** porque prioriza la **sensibilidad**, "
            "que es el criterio más importante cuando se busca no dejar casos positivos sin detectar."
        )
    elif supervised_entries:
        st.warning("Los modelos supervisados ya corren, pero aún no hay diagnóstico real para comparar sus métricas.")
    else:
        st.info("Ejecuta primero la pestaña de regresión para llenar esta sección.")

