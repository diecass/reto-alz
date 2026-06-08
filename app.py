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

# ==========================================================
# Configuración general
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
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RANDOM_STATE = 42

BASE_DIR = Path(r"C:\Users\dieca\OneDrive\Documentos\streamlit-app-ml")
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
        "logistic_regression_model.pkl",
        "modelo_regresion_logistica_alzheimer.pkl",
    ],
    "Random Forest": [
        "random_forest_model.pkl",
        "modelo_random_forest_mas_datos.pkl",
        "modelo_random_forest_pocos_datos.pkl",
    ],
    "Clustering": [
        "cluster_model.pkl",
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
# Utilidades de carga y validación
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


def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    schema = {}
    if df_ref is None:
        # Fallback manual; sensible defaults
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


def validate_and_prepare_csv(df_in: pd.DataFrame, schema: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = normalize_columns(df_in)
    issues = []

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_FEATURES + IDENTIFIER_COLS + [OPTIONAL_TARGET]]
    if missing:
        issues.append(f"Faltan columnas requeridas: {missing}")
    if extra:
        issues.append(f"Columnas extra que se ignorarán: {extra}")

    # Crear salida base con columnas esperadas
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
                # Si es entero-like, se aceptan 0/1 o enteros equivalentes
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
            # categóricas / binarias
            s = coerce_numeric(s)
            allowed = set(spec["options"])
            bad_cat = s.dropna()[~s.dropna().isin(list(allowed))]
            if len(bad_cat) > 0:
                issues.append(f"{c}: valores no permitidos {sorted(set(bad_cat.tolist()))[:10]}")
            out[c] = s.round(0)

    # Diagnóstico opcional
    if OPTIONAL_TARGET in df.columns:
        out[OPTIONAL_TARGET] = coerce_numeric(df[OPTIONAL_TARGET]).round(0)

    # Marcas de validez por fila
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


def to_model_input(df_prepared: pd.DataFrame) -> pd.DataFrame:
    x = df_prepared[EXPECTED_FEATURES].copy()
    # Mantener enteros donde aplica
    for c in INT_LIKE_COLS:
        if c in x.columns:
            x[c] = x[c].round(0).astype("Int64")
    # Numéricos a float
    for c in x.columns:
        if c not in INT_LIKE_COLS:
            x[c] = x[c].astype(float)
    return x


# ==========================================================
# Predicción y visualizaciones
# ==========================================================
def predict_supervised(model, x: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = model.predict(x)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[:, 1]
    return preds, probs


def get_rf_importance(model, x_columns: List[str]) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        out = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        return out
    except Exception:
        return None


def get_lr_coefficients(model) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        coef = model.named_steps["model"].coef_[0]
        out = pd.DataFrame({"feature": feature_names, "coef": coef})
        out["abs_coef"] = out["coef"].abs()
        out = out.sort_values("abs_coef", ascending=False)
        return out
    except Exception:
        return None


def predict_cluster(artifact, x: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    # Soporta: KMeans directo, dict con preprocessor + cluster_model, o pipeline completo
    if isinstance(artifact, dict):
        pre = artifact.get("preprocessor")
        cluster_model = artifact.get("cluster_model")
        if pre is not None and cluster_model is not None:
            x_t = pre.transform(x)
            labels = cluster_model.predict(x_t) if hasattr(cluster_model, "predict") else cluster_model.fit_predict(x_t)
            # Proyección PCA para viz
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            coords = pca.fit_transform(x_t)
            viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
            viz["Cluster"] = labels
            return labels, viz
    # pipeline o modelo directo
    if hasattr(artifact, "predict"):
        labels = artifact.predict(x)
        # intento de usar transform si existe preprocessor dentro de pipeline
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
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "manual_df" not in st.session_state:
    st.session_state["manual_df"] = pd.DataFrame(columns=EXPECTED_FEATURES)
if "validated_csv" not in st.session_state:
    st.session_state["validated_csv"] = None
if "csv_issues" not in st.session_state:
    st.session_state["csv_issues"] = []


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
# TAB 2: Captura manual
# ==========================================================
with tab2:
    st.subheader("Captura manual de un registro")
    st.write("Completa el formulario y agrega el registro a la cola de predicción.")

    with st.form("manual_form", clear_on_submit=False):
        left, right = st.columns(2)

        defaults = {}
        if reference_df is not None:
            for c in EXPECTED_FEATURES:
                if c in reference_df.columns:
                    if c in CATEGORICAL_COLS or c in BINARY_COLS or pd.api.types.is_integer_dtype(reference_df[c]):
                        defaults[c] = int(pd.Series(reference_df[c]).mode(dropna=True).iloc[0])
                    else:
                        defaults[c] = float(reference_df[c].median())
        else:
            defaults = {}

        def slider_or_number(col, key, spec):
            label = FEATURE_LABELS.get(key, key)
            if spec["type"] in {"binary", "categorical"}:
                opts = spec["options"]
                return col.selectbox(label, opts, index=opts.index(defaults.get(key, opts[0])) if defaults.get(key, opts[0]) in opts else 0)
            elif spec["type"] == "int":
                mn, mx = int(spec["min"]), int(spec["max"])
                return col.number_input(label, min_value=mn, max_value=mx, value=int(defaults.get(key, mn)), step=1)
            else:
                mn, mx = float(spec["min"]), float(spec["max"])
                return col.number_input(label, min_value=mn, max_value=mx, value=float(defaults.get(key, mn)), step=(mx - mn) / 100.0)

        record = {}
        cols = st.columns(3)
        idx = 0
        for feature in EXPECTED_FEATURES:
            c = cols[idx % 3]
            record[feature] = slider_or_number(c, feature, schema[feature])
            idx += 1

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
# TAB 3: Predicciones
# ==========================================================
with tab3:
    st.subheader("Ejecutar modelos")

    source = st.radio(
        "Fuente de datos",
        ["CSV validado", "Registros manuales", "Ambos"],
        horizontal=True,
    )

    selected_models = st.multiselect(
        "Selecciona los modelos a ejecutar",
        [m for m in ["Regresión logística", "Random Forest", "Clustering"] if m in models],
        default=[m for m in ["Regresión logística", "Random Forest", "Clustering"] if m in models][:2],
    )

    run_btn = st.button("Ejecutar predicción", type="primary", use_container_width=True)

    def get_input_data(source_choice: str) -> Optional[pd.DataFrame]:
        csv_df = st.session_state.get("validated_csv")
        man_df = st.session_state.get("manual_df")
        parts = []
        if source_choice in ["CSV validado", "Ambos"] and csv_df is not None:
            parts.append(csv_df[EXPECTED_FEATURES].copy())
        if source_choice in ["Registros manuales", "Ambos"] and man_df is not None and len(man_df) > 0:
            parts.append(man_df[EXPECTED_FEATURES].copy())
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

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

            results_frames = []
            for model_name in selected_models:
                st.markdown(f"### {model_name}")
                model_obj = models[model_name]

                if model_name in ["Regresión logística", "Random Forest"]:
                    preds, probs = predict_supervised(model_obj, x_in)
                    out = x_in.copy()
                    out[f"Pred_{model_name}"] = preds
                    if probs is not None:
                        out[f"Prob_{model_name}"] = probs
                    results_frames.append(out)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicción positiva", int(np.sum(preds == 1)))
                    c2.metric("Predicción negativa", int(np.sum(preds == 0)))
                    if probs is not None:
                        c3.metric("Probabilidad media", f"{np.mean(probs):.3f}")

                    st.dataframe(out.head(50), use_container_width=True)

                    if probs is not None:
                        fig = px.histogram(
                            pd.DataFrame({"Probabilidad": probs}),
                            x="Probabilidad",
                            nbins=20,
                            title=f"Distribución de probabilidades - {model_name}",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if model_name == "Random Forest":
                        imp = get_rf_importance(model_obj, EXPECTED_FEATURES)
                        if imp is not None:
                            fig_imp = px.bar(
                                imp.head(15).iloc[::-1],
                                x="importance",
                                y="feature",
                                orientation="h",
                                title="Importancia de variables - Random Forest",
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

                else:
                    labels, viz = predict_cluster(model_obj, x_in)
                    out = x_in.copy()
                    out["Cluster"] = labels
                    results_frames.append(out)

                    c1, c2 = st.columns(2)
                    c1.metric("Número de clusters detectados", int(pd.Series(labels).nunique()))
                    c2.metric("Cluster más frecuente", int(pd.Series(labels).mode().iloc[0]))

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

            if len(results_frames) > 0:
                merged = pd.concat(results_frames, axis=1)
                st.markdown("#### Descarga de resultados")
                st.download_button(
                    "Descargar resultados como CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_alzheimer.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ==========================================================
# TAB 4: Hallazgos
# ==========================================================
with tab4:
    st.subheader("Hallazgos más relevantes para el socio formador")

    st.markdown(
        """
        <div class="card">
            <span class="badge">Predicción</span>
            <span class="badge">Segmentación</span>
            <span class="badge">Validación de datos</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                La aplicación permite cargar datos, validar rangos y tipos, ejecutar modelos supervisados y no supervisados,
                y presentar resultados de forma clara para apoyar la toma de decisiones.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reference_df is not None:
        st.markdown("#### Resumen del dataset de referencia")
        summary_cols = EXPECTED_FEATURES[:10]
        summary = reference_df[summary_cols].describe().T[["mean", "std", "min", "max"]]
        st.dataframe(summary, use_container_width=True)

        fig_age = px.histogram(reference_df, x="Age", title="Distribución de edad en el dataset de referencia")
        st.plotly_chart(fig_age, use_container_width=True)

        if "Diagnosis" in reference_df.columns:
            diag = reference_df["Diagnosis"].value_counts().reset_index()
            diag.columns = ["Diagnosis", "Cantidad"]
            fig_diag = px.bar(diag, x="Diagnosis", y="Cantidad", title="Distribución de diagnóstico")
            st.plotly_chart(fig_diag, use_container_width=True)

    st.markdown("#### Recomendaciones de uso")
    st.write("- Usa **Random Forest** cuando quieras la mejor precisión práctica.")
    st.write("- Usa **Regresión logística** cuando necesites interpretación de variables.")
    st.write("- Usa **Clustering** para segmentar perfiles y visualizar grupos.")

st.caption("App preparada para Streamlit con carga de CSV, captura manual, validación y predicción.")
