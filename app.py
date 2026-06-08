import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
# VERSIONES ESPERADAS PARA COMPATIBILIDAD CON LOS .PKL
# ==========================================================
EXPECTED_VERSIONS = {
    "scikit-learn": "1.6.1",
    "numpy": "2.0.2",
    "pandas": "2.2.2",
    "scipy": "1.16.3",
}

CURRENT_VERSIONS = {
    "scikit-learn": __import__("sklearn").__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": __import__("scipy").__version__,
}

# ==========================================================
# APP CONFIG
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
BASE_DIR = Path(__file__).resolve().parent

# Si tu repo usa otra carpeta, cambia solo estos nombres.
MODELS_SUBDIR = ""
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
INT_LIKE_COLS = ["Age", "EducationLevel", "SystolicBP", "DiastolicBP"] + CATEGORICAL_COLS + BINARY_COLS

MODEL_FILENAMES = {
    "Regresión logística": ["modelo_regresion_logistica_alzheimer.pkl"],
    "Random Forest - GridSearch (pocos datos)": ["modelo_random_forest_pocos_datos.pkl"],
    "Random Forest - RandomizedSearch (más datos)": ["modelo_random_forest_mas_datos.pkl"],
    "Clustering": ["cluster_kmeans_full_model.pkl"],
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
# HELPERS
# ==========================================================
def resolve_existing_path(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def check_version_compatibility() -> List[str]:
    warnings = []
    for lib, expected in EXPECTED_VERSIONS.items():
        current = CURRENT_VERSIONS.get(lib, "desconocida")
        if current != expected:
            warnings.append(
                f"{lib}: instalada {current}, esperada {expected}. "
                "Esto puede causar problemas al cargar el .pkl."
            )
    return warnings


@st.cache_data(show_spinner=False)
def load_reference_dataset(data_path: str) -> Optional[pd.DataFrame]:
    p = Path(data_path)
    if not p.exists():
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}

    if df_ref is None:
        mins_maxs = {
            "Age": (60, 90),
            "BMI": (15.0, 40.0),
            "AlcoholConsumption": (0.0, 20.0),
            "PhysicalActivity": (0.0, 10.0),
            "DietQuality": (0.0, 10.0),
            "SleepQuality": (4.0, 10.0),
            "SystolicBP": (90, 180),
            "DiastolicBP": (60, 120),
            "CholesterolTotal": (150.0, 300.0),
            "CholesterolLDL": (50.0, 200.0),
            "CholesterolHDL": (20.0, 100.0),
            "CholesterolTriglycerides": (50.0, 400.0),
            "MMSE": (0.0, 30.0),
            "FunctionalAssessment": (0.0, 10.0),
            "ADL": (0.0, 10.0),
        }
        for c in EXPECTED_FEATURES:
            if c in CATEGORICAL_COLS:
                schema[c] = {"type": "categorical", "options": [0, 1] if c == "Gender" else [0, 1, 2, 3]}
            elif c in BINARY_COLS:
                schema[c] = {"type": "binary", "options": [0, 1]}
            elif c in INT_LIKE_COLS:
                schema[c] = {
                    "type": "int",
                    "min": mins_maxs.get(c, (0, 100))[0],
                    "max": mins_maxs.get(c, (0, 100))[1],
                }
            else:
                schema[c] = {
                    "type": "float",
                    "min": mins_maxs.get(c, (0.0, 1.0))[0],
                    "max": mins_maxs.get(c, (0.0, 1.0))[1],
                }
        return schema

    for c in EXPECTED_FEATURES:
        col = df_ref[c]
        if c in CATEGORICAL_COLS:
            opts = sorted([int(x) for x in col.dropna().unique().tolist()])
            schema[c] = {"type": "categorical", "options": opts}
        elif c in BINARY_COLS:
            schema[c] = {"type": "binary", "options": [0, 1]}
        elif pd.api.types.is_integer_dtype(col):
            schema[c] = {"type": "int", "min": int(col.min()), "max": int(col.max())}
        else:
            schema[c] = {"type": "float", "min": float(col.min()), "max": float(col.max())}
    return schema


def validate_and_prepare_csv(df_in: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
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
                non_int = s.dropna()[~np.isclose(s.dropna() % 1, 0)]
                if len(non_int) > 0:
                    issues.append(f"{c}: hay valores no enteros en filas {non_int.index.tolist()[:10]}")
                s = s.round(0)

            low, high = spec["min"], spec["max"]
            bad_range = s.dropna()[(s.dropna() < low) | (s.dropna() > high)]
            if len(bad_range) > 0:
                issues.append(f"{c}: valores fuera de rango [{low}, {high}] en filas {bad_range.index.tolist()[:10]}")

            out[c] = s
        else:
            s = coerce_numeric(s)
            allowed = set(spec["options"])
            bad_cat = s.dropna()[~s.dropna().isin(list(allowed))]
            if len(bad_cat) > 0:
                issues.append(f"{c}: valores no permitidos {sorted(set(bad_cat.tolist()))[:10]}")
            out[c] = s.round(0)

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


def safe_fit_metric(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


def plot_confusion_matrix_plotly(cm, title):
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
        width=700,
        height=500,
    )
    return fig


def to_model_input(df_prepared: pd.DataFrame) -> pd.DataFrame:
    x = df_prepared[EXPECTED_FEATURES].copy()
    for c in INT_LIKE_COLS:
        if c in x.columns:
            x[c] = x[c].round(0).astype("Int64")
    for c in x.columns:
        if c not in INT_LIKE_COLS:
            x[c] = x[c].astype(float)
    return x


def get_rf_importance(model) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        out = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return out
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
            x_t = artifact.named_steps["preprocess"].transform(x)
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(x_t)
            viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
            viz["Cluster"] = labels
            return labels, viz
        except Exception:
            viz = pd.DataFrame({"Cluster": labels})
            return labels, viz

    raise ValueError("No se pudo interpretar el artefacto de clustering.")


def load_models(models_dir: str) -> Dict[str, object]:
    loaded: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    for model_name, candidates in MODEL_FILENAMES.items():
        path = find_model_file(models_dir, candidates)
        if path is None:
            errors[model_name] = "No se encontró el archivo .pkl"
            continue

        try:
            loaded[model_name] = load_joblib_obj(str(path))
            loaded[f"{model_name}__path"] = str(path)
        except Exception as e:
            errors[model_name] = (
                f"{type(e).__name__}: {e}. "
                "Revisa que el entorno tenga exactamente las versiones compatibles "
                "con el entrenamiento del modelo."
            )

    st.session_state["model_load_errors"] = errors
    return loaded


# ==========================================================
# DEFAULT PATHS
# ==========================================================
ref_path_default = str(
    resolve_existing_path(
        DEFAULT_DATA_DIR / "alzheimer_dataset.xlsx",
        DEFAULT_DATA_DIR / "alzheimer_dataset.csv",
        BASE_DIR / "alzheimer_dataset.xlsx",
        BASE_DIR / "alzheimer_dataset.csv",
    ) or (DEFAULT_DATA_DIR / "alzheimer_dataset.xlsx")
)

models_dir_default = str(
    resolve_existing_path(
        DEFAULT_MODELS_DIR,
        BASE_DIR / "models",
        BASE_DIR / "artifacts" / "models",
    ) or DEFAULT_MODELS_DIR
)

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    models_dir = st.text_input("Carpeta de modelos", models_dir_default)
    reference_data_path = st.text_input("Dataset de referencia", ref_path_default)
    st.caption("Se usa para validar columnas, tipos y rangos.")

    st.markdown("### Versiones esperadas")
    for lib, ver in EXPECTED_VERSIONS.items():
        st.write(f"- {lib}: `{ver}`")

    st.markdown("### Versiones instaladas")
    for lib, ver in CURRENT_VERSIONS.items():
        st.write(f"- {lib}: `{ver}`")

    version_warnings = check_version_compatibility()
    if version_warnings:
        st.warning("Hay diferencias de versión que pueden afectar la lectura del .pkl.")
        with st.expander("Ver detalle"):
            for w in version_warnings:
                st.write(f"- {w}")
    else:
        st.success("Las versiones instaladas coinciden con las esperadas.")

    if st.button("Recargar modelos"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("### Modelos disponibles")
    st.write("- Regresión logística")
    st.write("- Random Forest (2 versiones)")
    st.write("- Clustering (KMeans)")

# ==========================================================
# LOAD DATA / MODELS
# ==========================================================
reference_df = load_reference_dataset(reference_data_path)
schema = build_schema_from_reference(reference_df)
models = load_models(models_dir)
model_load_errors = st.session_state.get("model_load_errors", {})

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">🧠 Alzheimer ML Dashboard</h1>
        <p style="margin:0.35rem 0 0 0; font-size:1.02rem; opacity:0.95;">
            Carga CSV, captura datos manualmente, valida tipos/rangos y ejecuta predicciones con modelos entrenados.
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
    st.warning("⚠️ No se encontró el dataset de referencia. La validación usará rangos genéricos.")
else:
    st.success(f"✅ Dataset de referencia cargado: {reference_df.shape[0]:,} filas × {reference_df.shape[1]:,} columnas")

missing_models = [
    m for m in [
        "Regresión logística",
        "Random Forest - GridSearch (pocos datos)",
        "Random Forest - RandomizedSearch (más datos)",
        "Clustering",
    ]
    if m not in models
]

if missing_models:
    st.error(f"❌ Faltan modelos: {missing_models}")
else:
    st.success("✅ Todos los modelos cargados correctamente")

if model_load_errors:
    with st.expander("Ver errores de carga de modelos"):
        for model_name, err in model_load_errors.items():
            st.write(f"**{model_name}**: {err}")

# ==========================================================
# SESSION STATE
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
if "prediction_output" not in st.session_state:
    st.session_state["prediction_output"] = None

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Cargar CSV",
    "✍️ Captura manual",
    "🤖 Predicciones",
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

            st.success(f"✅ Archivo cargado: {df_raw.shape[0]:,} filas × {df_raw.shape[1]:,} columnas")
            st.dataframe(df_raw.head(20), use_container_width=True)

            if issues:
                st.warning("⚠️ Observaciones de validación:")
                for i in issues:
                    st.write(f"- {i}")
            else:
                st.info("✅ No se detectaron problemas")

            valid_rate = validity["row_valid"].mean() * 100
            st.metric("Filas válidas", f"{valid_rate:.1f}%")

        except Exception as e:
            st.error(f"❌ Error al leer CSV: {e}")

# ==========================================================
# TAB 2: MANUAL
# ==========================================================
with tab2:
    st.subheader("Captura manual de registro")

    with st.form("manual_form", clear_on_submit=True):
        cols = st.columns(3)
        record = {}
        idx = 0

        for feature in EXPECTED_FEATURES:
            c = cols[idx % 3]
            label = FEATURE_LABELS.get(feature, feature)
            spec = schema[feature]

            if spec["type"] in {"binary", "categorical"}:
                opts = spec["options"]
                record[feature] = c.selectbox(label, opts, key=f"manual_{feature}")
            elif spec["type"] == "int":
                mn, mx = int(spec["min"]), int(spec["max"])
                record[feature] = c.number_input(label, min_value=mn, max_value=mx, value=mn, step=1, key=f"manual_{feature}")
            else:
                mn, mx = float(spec["min"]), float(spec["max"])
                step = (mx - mn) / 100.0 if mx > mn else 0.1
                record[feature] = c.number_input(label, min_value=mn, max_value=mx, value=mn, step=step, key=f"manual_{feature}")

            idx += 1

        submitted = st.form_submit_button("Agregar registro", use_container_width=True)

    if submitted:
        new_row = pd.DataFrame([record])
        st.session_state["manual_df"] = pd.concat([st.session_state["manual_df"], new_row], ignore_index=True)
        st.success("✅ Registro agregado")

    st.markdown("#### Registros almacenados")
    if len(st.session_state["manual_df"]) > 0:
        st.dataframe(st.session_state["manual_df"], use_container_width=True)

        if st.button("Limpiar registros"):
            st.session_state["manual_df"] = pd.DataFrame(columns=EXPECTED_FEATURES)
            st.rerun()
    else:
        st.info("Sin registros aún")

# ==========================================================
# TAB 3: PREDICCIONES
# ==========================================================
with tab3:
    st.subheader("Ejecutar modelos")

    st.markdown(
        """
        <div class="card">
            <span class="badge">Predicción clínica</span>
            <span class="badge">Sensibilidad priorizada</span>
            <span class="badge">CSV o manual</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Esta pestaña permite predecir aunque solo se agreguen datos manuales.
                Para los modelos supervisados, la decisión final se ajusta con un umbral configurable
                para priorizar la detección de casos positivos.
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

    available_model_names = [
        m for m in [
            "Regresión logística",
            "Random Forest - GridSearch (pocos datos)",
            "Random Forest - RandomizedSearch (más datos)",
            "Clustering",
        ] if m in models
    ]

    selected_models = st.multiselect(
        "Selecciona los modelos a ejecutar",
        available_model_names,
        default=available_model_names,
    )

    run_btn = st.button("Ejecutar predicción", type="primary", use_container_width=True)

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

    if run_btn:
        data_in = get_input_data(source)

        if data_in is None or len(data_in) == 0:
            st.error("❌ No hay datos. Carga CSV o agrega registros manuales.")
        elif len(selected_models) == 0:
            st.error("❌ Selecciona al menos un modelo.")
        else:
            x_in = to_model_input(data_in)

            st.success(f"✅ Datos listos: {x_in.shape[0]:,} filas × {x_in.shape[1]:,} variables")
            st.dataframe(x_in.head(20), use_container_width=True)

            # Output agregado sin duplicar columnas base
            results_df = x_in.copy()
            st.session_state["model_results"] = {}

            for model_name in selected_models:
                st.markdown(f"### {model_name}")
                model_obj = models[model_name]

                # -----------------------------
                # MODELOS SUPERVISADOS
                # -----------------------------
                if model_name in [
                    "Regresión logística",
                    "Random Forest - GridSearch (pocos datos)",
                    "Random Forest - RandomizedSearch (más datos)",
                ]:
                    preds, probs = predict_supervised(model_obj, x_in)

                    if probs is not None:
                        clinical_pred = np.where(probs >= sensitivity_threshold, 1, 0)
                    else:
                        clinical_pred = preds.copy()

                    pred_col = f"Pred_{model_name}"
                    results_df[pred_col] = preds

                    if probs is not None:
                        prob_col = f"Prob_{model_name}"
                        results_df[prob_col] = probs

                    clinical_col = f"Diagnóstico_clínico_{model_name}"
                    results_df[clinical_col] = np.where(
                        clinical_pred == 1,
                        "Tiene diagnóstico",
                        "No tiene diagnóstico",
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tiene diagnóstico", int(np.sum(clinical_pred == 1)))
                    c2.metric("No tiene diagnóstico", int(np.sum(clinical_pred == 0)))
                    c3.metric("Prob. media", f"{np.mean(probs):.3f}" if probs is not None else "N/A")

                    preview_cols = [pred_col]
                    if probs is not None:
                        preview_cols.append(prob_col)
                    preview_cols.append(clinical_col)

                    st.dataframe(
                        pd.concat([x_in.head(20), results_df[preview_cols].head(20)], axis=1),
                        use_container_width=True,
                    )

                    summary = pd.DataFrame({
                        "Resultado": ["Tiene diagnóstico", "No tiene diagnóstico"],
                        "Cantidad": [
                            int(np.sum(clinical_pred == 1)),
                            int(np.sum(clinical_pred == 0)),
                        ],
                    })
                    fig_summary = px.bar(summary, x="Resultado", y="Cantidad", title=f"Resumen - {model_name}")
                    st.plotly_chart(fig_summary, use_container_width=True)

                    if probs is not None:
                        fig = px.histogram(
                            pd.DataFrame({"Probabilidad": probs}),
                            x="Probabilidad",
                            nbins=20,
                            title=f"Distribución de probabilidades - {model_name}",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if "Random Forest" in model_name:
                        imp = get_rf_importance(model_obj)
                        if imp is not None:
                            fig_imp = px.bar(
                                imp.head(15).iloc[::-1],
                                x="importance",
                                y="feature",
                                orientation="h",
                                title=f"Top 15 variables - {model_name}",
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)

                    if model_name == "Regresión logística":
                        coef = get_lr_coefficients(model_obj)
                        if coef is not None:
                            top = coef.sort_values("abs_coef", ascending=False).head(15).copy()
                            top["dirección"] = np.where(top["coef"] >= 0, "Positivo", "Negativo")
                            fig_coef = px.bar(
                                top.iloc[::-1],
                                x="coef",
                                y="feature",
                                orientation="h",
                                color="dirección",
                                title="Top 15 coeficientes",
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)

                    st.session_state["model_results"][model_name] = {
                        "kind": "supervised",
                        "preds": preds,
                        "probs": probs,
                        "clinical_pred": clinical_pred,
                        "threshold": sensitivity_threshold,
                    }

                # -----------------------------
                # CLUSTERING
                # -----------------------------
                else:
                    labels, viz = predict_cluster(model_obj, x_in)

                    cluster_col = f"Cluster_{model_name}"
                    results_df[cluster_col] = labels

                    c1, c2 = st.columns(2)
                    c1.metric("Clusters", int(pd.Series(labels).nunique()))
                    c2.metric("Cluster más frecuente", int(pd.Series(labels).mode().iloc[0]))

                    st.dataframe(
                        pd.concat([x_in.head(20), results_df[[cluster_col]].head(20)], axis=1),
                        use_container_width=True,
                    )

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
                            title="Visualización PCA",
                            opacity=0.85,
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    st.session_state["model_results"][model_name] = {
                        "kind": "cluster",
                        "labels": labels,
                    }

            st.session_state["prediction_output"] = results_df

            st.markdown("#### Descargar resultados")
            st.download_button(
                "Descargar CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_alzheimer.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ==========================================================
# TAB 4: HALLAZGOS
# ==========================================================
with tab4:
    st.subheader("Resumen de resultados")

    results = st.session_state.get("model_results", {})
    output_df = st.session_state.get("prediction_output")

    if len(results) == 0:
        st.info("Ejecuta predicciones en la pestaña anterior para ver resultados.")
    else:
        st.success(f"✅ {len(results)} modelos ejecutados")

        supervised_rows = []

        for model_name, info in results.items():
            st.markdown(f"### {model_name}")

            if info.get("kind") == "supervised":
                c1, c2, c3 = st.columns(3)
                c1.metric("Positivos", int(np.sum(info.get("clinical_pred") == 1)))
                c2.metric("Negativos", int(np.sum(info.get("clinical_pred") == 0)))
                if info.get("probs") is not None:
                    c3.metric("Prob. media", f"{np.mean(info.get('probs')):.3f}")
                else:
                    c3.metric("Prob. media", "N/A")

                # Métricas solo si existe Diagnosis real en el CSV validado
                y_true = None
                if (
                    st.session_state.get("uploaded_df") is not None
                    and st.session_state.get("validated_csv") is not None
                    and OPTIONAL_TARGET in st.session_state["uploaded_df"].columns
                    and len(st.session_state["uploaded_df"]) == len(st.session_state["validated_csv"])
                ):
                    y_true = (
                        pd.to_numeric(st.session_state["uploaded_df"][OPTIONAL_TARGET], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .values
                    )

                if y_true is not None and len(y_true) == len(info.get("clinical_pred")):
                    metrics, cm = safe_fit_metric(y_true, info.get("clinical_pred"), info.get("probs"))
                    supervised_rows.append({
                        "Modelo": model_name,
                        "Accuracy": metrics["accuracy"],
                        "Precision": metrics["precision"],
                        "Sensibilidad": metrics["recall"],
                        "Especificidad": cm.ravel()[0] / (cm.ravel()[0] + cm.ravel()[1]) if (cm.ravel()[0] + cm.ravel()[1]) > 0 else 0.0,
                        "F1": metrics["f1"],
                        "AUC": metrics["auc"],
                    })

                    st.plotly_chart(
                        plot_confusion_matrix_plotly(cm, f"Matriz de confusión - {model_name}"),
                        use_container_width=True,
                    )

                    if info.get("probs") is not None:
                        try:
                            fpr, tpr, _ = roc_curve(y_true, info.get("probs"))
                            roc_fig = go.Figure()
                            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
                            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash")))
                            roc_fig.update_layout(title=f"Curva ROC - {model_name}", xaxis_title="FPR", yaxis_title="TPR")
                            st.plotly_chart(roc_fig, use_container_width=True)
                        except Exception:
                            st.info("No se pudo generar la curva ROC con este conjunto de datos.")

                    st.write(
                        f"Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}."
                    )
                else:
                    st.info("No existe una columna real de diagnóstico para este conjunto, así que solo se muestra el resultado predictivo.")

            elif info.get("kind") == "cluster":
                labels = info.get("labels")
                c1, c2 = st.columns(2)
                c1.metric("Clusters", int(pd.Series(labels).nunique()))
                c2.metric("Tamaño promedio", f"{len(labels) / pd.Series(labels).nunique():.0f}")

            st.markdown("---")

        if supervised_rows:
            st.markdown("### Comparación de modelos supervisados")
            df_supervised = pd.DataFrame(supervised_rows)

            df_supervised = df_supervised.sort_values(
                by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
                ascending=False,
            )

            st.dataframe(df_supervised, use_container_width=True, hide_index=True)

            metric_cols = [c for c in ["Accuracy", "Precision", "Sensibilidad", "Especificidad", "F1", "AUC"] if c in df_supervised.columns]
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

            best_row = df_supervised.iloc[0]
            st.success(
                f"**Mejor modelo seleccionado: {best_row['Modelo']}**. "
                f"Se eligió por tener la mayor sensibilidad, con F1={best_row['F1']:.3f}."
            )

        if output_df is not None:
            st.markdown("### Vista rápida del archivo de salida")
            st.dataframe(output_df.head(20), use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("### 📚 Información")
st.markdown(f"**Modelos cargados:** {len([k for k in models.keys() if not k.endswith('__path')])}/4")
st.markdown(f"**Variables:** {len(EXPECTED_FEATURES)}")
st.markdown(f"**Carpeta de modelos actual:** `{models_dir}`")
st.markdown("**Versiones esperadas para leer los .pkl:**")
st.write(EXPECTED_VERSIONS)
