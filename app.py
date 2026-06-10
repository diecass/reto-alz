import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
            background: linear-gradient(135deg, #0f172a 0%, #1e40af 45%, #38bdf8 100%);
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.20);
            margin-bottom: 1rem;
        }
        .card {
            background: white;
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
        }
        .small-note {color: #475569; font-size: 0.92rem;}
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.8rem;
            margin-right: 0.35rem;
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

BLUE_SEQ = px.colors.sequential.Blues


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
    loaded: Dict[str, object] = {}
    for model_name, candidates in MODEL_FILENAMES.items():
        path = find_model_file(models_dir, candidates)
        if path is not None:
            loaded[model_name] = load_joblib_obj(str(path))
            loaded[f"{model_name}__path"] = str(path)
    return loaded


def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict]:
    schema: Dict[str, Dict] = {}

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
                lo, hi = mins_maxs.get(c, (0, 100))
                schema[c] = {"type": "int", "min": lo, "max": hi}
            else:
                lo, hi = mins_maxs.get(c, (0.0, 1.0))
                schema[c] = {"type": "float", "min": lo, "max": hi}
        return schema

    for c in EXPECTED_FEATURES:
        col = df_ref[c]
        if c in CATEGORICAL_COLS:
            values = sorted([int(x) for x in pd.Series(col).dropna().unique().tolist()])
            schema[c] = {"type": "categorical", "options": values}
        elif c in BINARY_COLS:
            schema[c] = {"type": "binary", "options": [0, 1]}
        elif pd.api.types.is_integer_dtype(col):
            schema[c] = {"type": "int", "min": int(col.min()), "max": int(col.max())}
        else:
            schema[c] = {"type": "float", "min": float(col.min()), "max": float(col.max())}
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
        valid = out[c].notna()
        if spec["type"] in {"float", "int"}:
            valid &= (out[c].astype(float) >= spec["min"]) & (out[c].astype(float) <= spec["max"])
        else:
            valid &= out[c].isin(spec["options"])
        validity[f"{c}_valid"] = valid
    validity["row_valid"] = validity[[f"{c}_valid" for c in EXPECTED_FEATURES]].all(axis=1)

    return out, validity, issues


def to_model_input(df_prepared: pd.DataFrame) -> pd.DataFrame:
    x = df_prepared[EXPECTED_FEATURES].copy()
    for c in INT_LIKE_COLS:
        if c in x.columns:
            x[c] = x[c].round(0).astype("Int64")
    for c in x.columns:
        if c not in INT_LIKE_COLS:
            x[c] = x[c].astype(float)
    return x


def safe_default_value(col: str, reference_df: Optional[pd.DataFrame], spec: Dict) -> Union[int, float]:
    if reference_df is not None and col in reference_df.columns and reference_df[col].notna().any():
        series = reference_df[col]
        if spec["type"] in {"binary", "categorical"} or pd.api.types.is_integer_dtype(series):
            mode = series.mode(dropna=True)
            if len(mode) > 0:
                return int(mode.iloc[0])
        try:
            return float(series.median())
        except Exception:
            pass

    if spec["type"] in {"binary", "categorical"}:
        return int(spec["options"][0])
    if spec["type"] == "int":
        return int(spec["min"])
    return float(spec["min"])


def build_model_input_form(reference_df: Optional[pd.DataFrame], schema: Dict[str, Dict]) -> pd.DataFrame:
    defaults = {c: safe_default_value(c, reference_df, schema[c]) for c in EXPECTED_FEATURES}
    record: Dict[str, Union[int, float]] = {}

    cols = st.columns(3)
    idx = 0
    for feature in EXPECTED_FEATURES:
        c = cols[idx % 3]
        spec = schema[feature]
        label = FEATURE_LABELS.get(feature, feature)

        if spec["type"] in {"binary", "categorical"}:
            opts = spec["options"]
            default_val = defaults.get(feature, opts[0])
            index = opts.index(default_val) if default_val in opts else 0
            record[feature] = c.selectbox(label, opts, index=index)
        elif spec["type"] == "int":
            mn, mx = int(spec["min"]), int(spec["max"])
            record[feature] = c.number_input(label, min_value=mn, max_value=mx, value=int(defaults.get(feature, mn)), step=1)
        else:
            mn, mx = float(spec["min"]), float(spec["max"])
            step = max((mx - mn) / 100.0, 0.01)
            record[feature] = c.number_input(label, min_value=mn, max_value=mx, value=float(defaults.get(feature, mn)), step=step)
        idx += 1

    return pd.DataFrame([record])


# ==========================================================
# Predicción y visualizaciones
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


def get_rf_importance(model) -> Optional[pd.DataFrame]:
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    except Exception:
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            importances = model.named_steps["model"].feature_importances_
            return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
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
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            coef = model.named_steps["model"].coef_[0]
            out = pd.DataFrame({"feature": feature_names, "coef": coef})
            out["abs_coef"] = out["coef"].abs()
            return out.sort_values("abs_coef", ascending=False)
        except Exception:
            return None


def evaluate_binary_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
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


def interpret_model_results(model_name: str, metrics: Dict) -> str:
    tn, fp, fn, tp = metrics["cm"].ravel()

    parts = [
        f"**{model_name}**",
        (
            f"Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
            f"Sensibilidad/Recall={metrics['recall']:.3f}, Especificidad={metrics['specificity']:.3f}, "
            f"F1={metrics['f1']:.3f}"
            + (f", AUC={metrics['auc']:.3f}." if not np.isnan(metrics["auc"]) else ".")
        ),
        f"Matriz de confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}.",
    ]

    if metrics["recall"] >= metrics["precision"]:
        parts.append("El modelo prioriza la detección de positivos, lo cual es útil cuando conviene no dejar casos reales sin identificar.")
    else:
        parts.append("El modelo es más conservador al marcar positivos, reduciendo falsos positivos pero pudiendo perder algunos casos reales.")

    if fn < fp:
        parts.append("Los falsos negativos son menores que los falsos positivos, algo favorable en un contexto médico.")
    elif fn > fp:
        parts.append("Hay más falsos negativos que falsos positivos; convendría revisar el umbral para mejorar la sensibilidad.")
    else:
        parts.append("Los errores positivos y negativos están equilibrados.")

    return " ".join(parts)


def plot_confusion_matrix(cm: np.ndarray, title: str) -> go.Figure:
    tn, fp, fn, tp = cm.ravel()
    z = [[tn, fp], [fn, tp]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicho 0", "Predicho 1"],
            y=["Real 0", "Real 1"],
            text=z,
            texttemplate="%{text}",
            colorscale=BLUE_SEQ,
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


def get_cluster_steps(artifact) -> Tuple[Optional[object], Optional[object]]:
    """
    Devuelve (preprocessor, cluster_model) desde:
    - dict con llaves comunes
    - pipeline con named_steps
    - estimador directo
    """
    preprocessor = None
    cluster_model = None

    if isinstance(artifact, dict):
        for k in ("preprocess", "preprocessor", "pipeline_preprocessor"):
            if k in artifact:
                preprocessor = artifact[k]
                break
        for k in ("cluster_model", "model", "kmeans", "estimator"):
            if k in artifact:
                cluster_model = artifact[k]
                break

    elif hasattr(artifact, "named_steps"):
        steps = getattr(artifact, "named_steps")
        for k in ("preprocess", "preprocessor"):
            if k in steps:
                preprocessor = steps[k]
                break
        for name, step in steps.items():
            if hasattr(step, "cluster_centers_") or hasattr(step, "predict") and "cluster" in name.lower():
                cluster_model = step
                break
        if cluster_model is None:
            for name, step in steps.items():
                if hasattr(step, "cluster_centers_"):
                    cluster_model = step
                    break
        if cluster_model is None and hasattr(artifact, "cluster_centers_"):
            cluster_model = artifact

    else:
        cluster_model = artifact if hasattr(artifact, "predict") else None

    return preprocessor, cluster_model


def predict_cluster(artifact, x: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Retorna:
    - labels
    - viz (PC1, PC2, Cluster)
    - centroids_viz (si existe)
    - transformed dataframe (si se puede construir)
    """
    preprocessor, cluster_model = get_cluster_steps(artifact)

    if hasattr(artifact, "predict") and cluster_model is None:
        cluster_model = artifact

    if preprocessor is not None:
        x_t = preprocessor.transform(x)
        try:
            feature_names = preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(x_t, columns=feature_names, index=x.index)
        except Exception:
            transformed_df = pd.DataFrame(x_t, index=x.index)
    else:
        x_t = x.to_numpy()
        transformed_df = x.copy()

    # Etiquetas
    if cluster_model is not None and hasattr(cluster_model, "predict"):
        labels = cluster_model.predict(x_t) if preprocessor is not None and cluster_model is not artifact else cluster_model.predict(x)
    elif hasattr(artifact, "predict"):
        labels = artifact.predict(x)
    else:
        raise ValueError("No se pudo interpretar el artefacto de clustering.")

    labels = np.asarray(labels)

    # PCA para visualización
    n_components = 2 if x_t.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x_t)

    if n_components == 1:
        viz = pd.DataFrame({"PC1": coords[:, 0], "PC2": np.zeros(len(coords)), "Cluster": labels})
    else:
        viz = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels})

    centroids_viz = None
    if cluster_model is not None and hasattr(cluster_model, "cluster_centers_"):
        try:
            centroid_coords = pca.transform(np.asarray(cluster_model.cluster_centers_))
            if centroid_coords.shape[1] == 1:
                centroids_viz = pd.DataFrame({
                    "PC1": centroid_coords[:, 0],
                    "PC2": np.zeros(len(centroid_coords)),
                    "Cluster": np.arange(len(centroid_coords)),
                })
            else:
                centroids_viz = pd.DataFrame({
                    "PC1": centroid_coords[:, 0],
                    "PC2": centroid_coords[:, 1],
                    "Cluster": np.arange(len(centroid_coords)),
                })
        except Exception:
            centroids_viz = None

    return labels, viz, centroids_viz, transformed_df


def cluster_feature_medians(x: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = x.copy()
    df["Cluster"] = labels
    medians = df.groupby("Cluster")[EXPECTED_FEATURES].median(numeric_only=True)
    return df, medians


def top_cluster_features(medians: pd.DataFrame, top_n: int = 10) -> List[str]:
    if 0 in medians.index and 1 in medians.index:
        diffs = (medians.loc[0] - medians.loc[1]).abs().sort_values(ascending=False)
        return diffs.head(top_n).index.tolist()
    return medians.mean(axis=0).sort_values(ascending=False).head(top_n).index.tolist()


def plot_cluster_pca(viz: pd.DataFrame, centroids: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = px.scatter(
        viz,
        x="PC1",
        y="PC2",
        color=viz["Cluster"].astype(str),
        title="Visualización PCA de los grupos",
        opacity=0.82,
        color_discrete_sequence=BLUE_SEQ,
    )
    if centroids is not None and len(centroids) > 0:
        fig.add_trace(
            go.Scatter(
                x=centroids["PC1"],
                y=centroids["PC2"],
                mode="markers+text",
                text=[f"C{int(c)}" for c in centroids["Cluster"]],
                textposition="top center",
                marker=dict(size=16, symbol="x", color="#0f172a", line=dict(width=2, color="#0f172a")),
                name="Centroides",
                showlegend=True,
            )
        )
    fig.update_layout(height=560)
    return fig


def plot_cluster_distribution(labels: np.ndarray) -> Tuple[go.Figure, pd.DataFrame]:
    counts = pd.Series(labels).value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Cantidad"]

    fig = px.pie(
        counts,
        names="Cluster",
        values="Cantidad",
        title="Distribución de grupos",
        color="Cluster",
        color_discrete_sequence=BLUE_SEQ,
        hole=0.35,
    )
    return fig, counts


def plot_cluster_feature_bars(medians: pd.DataFrame, cluster_id: int, features: List[str]) -> go.Figure:
    plot_data = pd.DataFrame({
        "Variable": [FEATURE_LABELS.get(f, f) for f in features],
        "Mediana": [float(medians.loc[cluster_id, f]) if cluster_id in medians.index and f in medians.columns else np.nan for f in features],
    }).dropna()

    plot_data = plot_data.sort_values("Mediana", ascending=True)

    fig = px.bar(
        plot_data,
        x="Mediana",
        y="Variable",
        orientation="h",
        title=f"Cluster {cluster_id} — variables más distintivas por mediana",
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(height=480)
    return fig


# ==========================================================
# Estado de sesión
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
# Carga inicial
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
        <h1 style="margin:0;">🧠 Alzheimer ML Dashboard</h1>
        <p style="margin:0.35rem 0 0 0; font-size:1.02rem; opacity:0.95;">
            Carga CSV, captura datos manualmente, valida tipos/rangos y ejecuta predicciones con una interfaz clara y profesional.
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Cargar CSV",
    "✍️ Captura manual",
    "🤖 Predicciones",
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
# TAB 2: Captura manual
# ==========================================================
with tab2:
    st.subheader("Captura manual de un registro")
    st.write("Completa el formulario y agrega el registro a la cola de predicción.")

    with st.form("manual_form", clear_on_submit=False):
        manual_df_preview = build_model_input_form(reference_df, schema)
        submitted = st.form_submit_button("Agregar registro", use_container_width=True)

    if submitted:
        st.session_state["manual_df"] = pd.concat([st.session_state["manual_df"], manual_df_preview], ignore_index=True)
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

    source = st.radio("Fuente de datos", ["CSV validado", "Registros manuales", "Ambos"], horizontal=True)

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

                    out = x_in.copy()
                    out[f"Pred_{model_name}"] = preds
                    if probs is not None:
                        out[f"Prob_{model_name}"] = probs
                    out[f"Diagnóstico_clínico_{model_name}"] = np.where(
                        clinical_pred == 1,
                        "Tiene diagnóstico",
                        "No tiene diagnóstico",
                    )

                    results_frames.append(out)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tiene diagnóstico", int(np.sum(clinical_pred == 1)))
                    c2.metric("No tiene diagnóstico", int(np.sum(clinical_pred == 0)))
                    c3.metric("Probabilidad media", f"{np.mean(probs):.3f}" if probs is not None else "N/A")

                    st.dataframe(out.head(50), use_container_width=True)

                    # Guardar y_true solo cuando el CSV validado contiene la columna Diagnosis
                    y_true = None
                    if source == "CSV validado":
                        uploaded_df = st.session_state.get("uploaded_df")
                        validated_csv = st.session_state.get("validated_csv")
                        if (
                            uploaded_df is not None
                            and validated_csv is not None
                            and OPTIONAL_TARGET in uploaded_df.columns
                            and len(uploaded_df) == len(validated_csv)
                        ):
                            y_true = pd.to_numeric(uploaded_df[OPTIONAL_TARGET], errors="coerce").fillna(0).astype(int).values

                    # Métricas y descripción
                    if y_true is not None and len(y_true) == len(clinical_pred):
                        metrics = evaluate_binary_model(y_true, clinical_pred, probs)
                        st.plotly_chart(plot_confusion_matrix(metrics["cm"], f"Matriz de confusión - {model_name}"), use_container_width=True)

                        if probs is not None and len(np.unique(y_true)) > 1:
                            roc_fig = go.Figure()
                            fpr, tpr, _ = roc_curve(y_true, probs)
                            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name, line=dict(color="#2563eb", width=3)))
                            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash", color="#64748b")))
                            roc_fig.update_layout(
                                title=f"Curva ROC - {model_name}",
                                xaxis_title="FPR",
                                yaxis_title="TPR",
                                height=460,
                            )
                            st.plotly_chart(roc_fig, use_container_width=True)

                        st.write(interpret_model_results(model_name, metrics))
                    else:
                        pred_series = pd.Series(clinical_pred)
                        positivos = int((pred_series == 1).sum())
                        negativos = int((pred_series == 0).sum())
                        st.info(
                            "No existe una columna real de diagnóstico para comparar, así que esta salida se interpreta como una estimación clínica."
                        )
                        st.write(
                            f"Este modelo marcó {positivos} casos como positivos y {negativos} como negativos. "
                            f"Sin diagnóstico real, la lectura debe centrarse en la distribución de casos y en la coherencia del patrón de salida."
                        )

                    st.session_state["model_results"][model_name] = {
                        "kind": "supervised",
                        "preds": preds,
                        "probs": probs,
                        "clinical_pred": clinical_pred,
                        "threshold": sensitivity_threshold,
                        "y_true": y_true,
                    }

                    if "Random Forest" in model_name:
                        imp = get_rf_importance(model_obj)
                        if imp is not None:
                            fig_imp = px.bar(
                                imp.head(15).iloc[::-1],
                                x="importance",
                                y="feature",
                                orientation="h",
                                title=f"Importancia de variables - {model_name}",
                                color_discrete_sequence=["#2563eb"],
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
                                color_discrete_sequence=["#2563eb", "#60a5fa"],
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)

                    st.markdown("---")

            if len(results_frames) > 0:
                merged = pd.concat(results_frames, axis=1)

                st.markdown("#### Comparación rápida entre modelos")
                rf_models_present = [
                    m for m in [
                        "Random Forest - GridSearch (pocos datos)",
                        "Random Forest - RandomizedSearch (más datos)",
                    ] if m in selected_models
                ]

                if len(rf_models_present) == 2:
                    compare_rows = []
                    for rf_name in rf_models_present:
                        pred_col = f"Pred_{rf_name}"
                        prob_col = f"Prob_{rf_name}"
                        diag_col = f"Diagnóstico_clínico_{rf_name}"
                        compare_rows.append({
                            "Modelo": rf_name,
                            "Positivos predichos": int((merged[diag_col] == "Tiene diagnóstico").sum()) if diag_col in merged.columns else int((merged[pred_col] == 1).sum()),
                            "Negativos predichos": int((merged[diag_col] == "No tiene diagnóstico").sum()) if diag_col in merged.columns else int((merged[pred_col] == 0).sum()),
                            "Probabilidad media": float(merged[prob_col].mean()) if prob_col in merged.columns else np.nan,
                        })

                    compare = pd.DataFrame(compare_rows)
                    st.dataframe(compare, use_container_width=True, hide_index=True)

                    fig_cmp = px.bar(
                        compare.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"),
                        x="Modelo",
                        y="Valor",
                        color="Métrica",
                        barmode="group",
                        title="Comparación entre los dos Random Forest",
                        color_discrete_sequence=BLUE_SEQ,
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                st.markdown("#### Descarga de resultados")
                st.download_button(
                    "Descargar resultados como CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_alzheimer.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ==========================================================
# TAB 4: Grupos
# ==========================================================
with tab4:
    st.subheader("Grupos")

    st.markdown(
        """
        <div class="card">
            <span class="badge">Clustering</span>
            <span class="badge">Sin diagnóstico</span>
            <span class="badge">Perfiles</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Esta pestaña quedó solo para grupos. El clustering se ejecuta únicamente con CSV validado,
                porque aquí sí se trabaja con una base completa y no con registros manuales o combinados.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cluster_source = st.radio(
        "Fuente para grupos",
        ["CSV validado", "Registros manuales", "Ambos"],
        horizontal=True,
        help="Para esta pestaña se usa solo CSV validado.",
    )

    run_cluster = st.button("Calcular grupos", type="primary", use_container_width=True)

    if run_cluster:
        if cluster_source != "CSV validado":
            st.warning("El clustering solo se calcula con CSV validado. Con registros manuales o combinados no se ejecuta esta sección.")
        elif "Clustering" not in models:
            st.error("No se encontró el modelo de clustering en la carpeta de modelos.")
        else:
            csv_df = st.session_state.get("validated_csv")
            if csv_df is None or len(csv_df) == 0:
                st.error("Primero carga un CSV válido en la pestaña de carga.")
            else:
                x_in = to_model_input(csv_df)
                model_obj = models["Clustering"]

                labels, viz, centroids_viz, transformed_df = predict_cluster(model_obj, x_in)
                _, cluster_counts = plot_cluster_distribution(labels)
                df_with_labels, medians = cluster_feature_medians(x_in, labels)

                out = x_in.copy()
                out["Cluster"] = labels

                c1, c2, c3 = st.columns(3)
                c1.metric("Número de grupos detectados", int(pd.Series(labels).nunique()))
                c2.metric("Grupo más frecuente", int(pd.Series(labels).mode().iloc[0]))
                c3.metric("Tamaño promedio por grupo", f"{pd.Series(labels).value_counts().mean():.1f}")

                st.dataframe(out.head(50), use_container_width=True)

                fig_pie, counts = plot_cluster_distribution(labels)
                st.plotly_chart(fig_pie, use_container_width=True)

                # Distribución en tabla para que la división quede visible en esta pestaña
                st.dataframe(counts, use_container_width=True, hide_index=True)

                fig_scatter = plot_cluster_pca(viz, centroids_viz)
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.info(
                    "El clustering no predice diagnóstico; agrupa pacientes con características similares. "
                    "Las centroides ayudan a ver que el método sí corresponde a K-Means, y el PCA resume los patrones en dos dimensiones."
                )

                st.session_state["model_results"]["Clustering"] = {
                    "kind": "cluster",
                    "labels": labels,
                }
                st.session_state["cluster_results"] = {
                    "x": x_in,
                    "labels": labels,
                    "viz": viz,
                    "centroids_viz": centroids_viz,
                    "medians": medians,
                    "counts": counts,
                }
    else:
        st.info("Presiona 'Calcular grupos' para generar la salida de clustering.")


# ==========================================================
# TAB 5: Hallazgos
# ==========================================================
with tab5:
    st.subheader("Hallazgos más relevantes")

    st.markdown(
        """
        <div class="card">
            <span class="badge">Interpretación automática</span>
            <span class="badge">Comparación clínica</span>
            <span class="badge">Sensibilidad priorizada</span>
            <p style="margin-top:0.8rem; margin-bottom:0;">
                Esta sección resume los modelos ejecutados, priorizando la sensibilidad para decidir cuál es el mejor en un contexto clínico.
                Las gráficas de ROC y matriz de confusión se muestran en la pestaña de predicciones, justo después de cada modelo seleccionado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = st.session_state.get("model_results", {})

    # ==========================================================
    # RESUMEN DE MODELOS SUPERVISADOS
    # ==========================================================
    supervised_rows = []
    supervised_available = False

    for model_name, info in results.items():
        if info.get("kind") != "supervised":
            continue

        supervised_available = True
        y_pred = info.get("clinical_pred", info.get("preds"))
        y_prob = info.get("probs")
        y_true = info.get("y_true")

        if y_true is not None and len(y_true) == len(y_pred):
            metrics = evaluate_binary_model(y_true, y_pred, y_prob)
            supervised_rows.append({
                "Modelo": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Sensibilidad": metrics["recall"],
                "Especificidad": metrics["specificity"],
                "F1": metrics["f1"],
                "AUC": metrics["auc"],
            })
        else:
            supervised_rows.append({
                "Modelo": model_name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Sensibilidad": np.nan,
                "Especificidad": np.nan,
                "F1": np.nan,
                "AUC": np.nan,
            })

    if supervised_available and len(supervised_rows) > 0:
        st.markdown("### Comparación de modelos supervisados")
        df_supervised = pd.DataFrame(supervised_rows)

        if df_supervised["Sensibilidad"].notna().any():
            df_supervised = df_supervised.sort_values(
                by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
                ascending=False,
            )

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

        metric_cols = [c for c in ["Precision", "Sensibilidad", "Especificidad", "F1", "Accuracy", "AUC"] if c in df_supervised.columns]
        if metric_cols:
            plot_df = df_supervised.melt(id_vars="Modelo", value_vars=metric_cols, var_name="Métrica", value_name="Valor")
            fig_metrics = px.bar(
                plot_df,
                x="Modelo",
                y="Valor",
                color="Métrica",
                barmode="group",
                title="Comparación de desempeño entre modelos supervisados",
                color_discrete_sequence=BLUE_SEQ,
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        if df_supervised["Sensibilidad"].notna().any():
            best_row = df_supervised.sort_values(
                by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
                ascending=False,
            ).iloc[0]

            st.success(
                f"Mejor modelo seleccionado: **{best_row['Modelo']}**. "
                f"Se eligió porque obtuvo la mayor sensibilidad, que es el criterio principal en este proyecto. "
                f"Además, mantiene un F1 de {best_row['F1']:.3f}, lo que indica un balance razonable entre precisión y sensibilidad."
            )

        st.markdown("### Resumen por modelo")
        for _, row in df_supervised.iterrows():
            model_name = row["Modelo"]
            info = results[model_name]
            y_pred = info.get("clinical_pred", info.get("preds"))
            y_prob = info.get("probs")
            y_true = info.get("y_true")

            st.subheader(model_name)

            if y_true is not None and len(y_true) == len(y_pred):
                metrics = evaluate_binary_model(y_true, y_pred, y_prob)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                c2.metric("Precision", f"{metrics['precision']:.3f}")
                c3.metric("Sensibilidad", f"{metrics['recall']:.3f}")
                c4.metric("Especificidad", f"{metrics['specificity']:.3f}")
                c5.metric("F1", f"{metrics['f1']:.3f}")
                st.write(interpret_model_results(model_name, metrics))
            else:
                pred_series = pd.Series(y_pred)
                positivos = int((pred_series == 1).sum())
                negativos = int((pred_series == 0).sum())

                c1, c2 = st.columns(2)
                c1.metric("Tiene diagnóstico", positivos)
                c2.metric("No tiene diagnóstico", negativos)

                st.write(
                    f"Este modelo marcó {positivos} casos como positivos y {negativos} como negativos. "
                    f"Como no hay diagnóstico real para comparar, esta salida se interpreta como una estimación clínica."
                )

            st.markdown("---")
    else:
        st.info("No se han ejecutado modelos supervisados todavía.")

    # ==========================================================
    # HALLAZGOS DE CLUSTERING
    # ==========================================================
    cluster_results = st.session_state.get("cluster_results", {})
    if cluster_results:
        st.markdown("### Hallazgos de clustering")

        labels = np.asarray(cluster_results.get("labels"))
        x_in = cluster_results.get("x")
        medians = cluster_results.get("medians")
        counts = cluster_results.get("counts")

        if x_in is not None and medians is not None and len(labels) == len(x_in):
            if 0 in medians.index and 1 in medians.index:
                top_features = top_cluster_features(medians, top_n=10)
                diff_df = pd.DataFrame({
                    "Variable": [FEATURE_LABELS.get(f, f) for f in top_features],
                    "Diferencia de medianas": [abs(float(medians.loc[0, f] - medians.loc[1, f])) for f in top_features],
                }).sort_values("Diferencia de medianas", ascending=False)

                st.markdown("**Variables con mayor separación entre Cluster 0 y Cluster 1**")
                st.dataframe(diff_df, use_container_width=True, hide_index=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(plot_cluster_feature_bars(medians, 0, top_features), use_container_width=True)
                with col_r:
                    st.plotly_chart(plot_cluster_feature_bars(medians, 1, top_features), use_container_width=True)

                st.write(
                    "Estas gráficas muestran las medianas de las variables que más diferencian a los dos clusters principales. "
                    "Así se entiende mejor qué perfil caracteriza a cada grupo."
                )
            else:
                st.info(
                    "El modelo de clustering detectó más de dos grupos o una distribución distinta a 0 y 1. "
                    "Se mantiene la visualización general, pero la comparación detallada de variables está optimizada para los clusters 0 y 1."
                )

            if counts is not None:
                st.markdown("**Distribución resumida de clusters**")
                st.dataframe(counts, use_container_width=True, hide_index=True)

    else:
        st.info("No se han ejecutado grupos todavía.")

    # ==========================================================
    # CONCLUSIÓN EJECUTIVA
    # ==========================================================
    st.markdown("### Conclusión ejecutiva")
    if supervised_available and len(supervised_rows) > 0:
        df_exec = pd.DataFrame(supervised_rows)

        if df_exec["Sensibilidad"].notna().any():
            best_exec = df_exec.sort_values(
                by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
                ascending=False,
            ).iloc[0]

            st.success(
                f"""
En términos clínicos, el modelo más conveniente es **{best_exec['Modelo']}** porque prioriza la **sensibilidad**,
que es el criterio más importante cuando se busca no dejar casos positivos sin detectar.

Su desempeño general también es sólido en F1 y especificidad, por lo que mantiene un equilibrio aceptable entre
detectar correctamente pacientes con posible diagnóstico y evitar errores excesivos.
"""
            )
        else:
            st.warning(
                "No fue posible calcular una comparación clínica real porque los datos cargados no contienen diagnóstico verdadero."
            )
    else:
        st.info("Ejecuta al menos un modelo supervisado para generar una conclusión ejecutiva.")
