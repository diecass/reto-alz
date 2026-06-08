import os
from pathlib import Pathfrom typing 
import Dict, List, Tuple, Optional

import joblibimport numpy as npimport pandas as pdimport streamlit as stimport plotly.express as pximport plotly.graph_objects as gofrom sklearn.decomposition import PCA

st.set_page_config(page_title="Alzheimer ML App",page_icon="🧠",layout="wide",initial_sidebar_state="expanded",)

st.markdown(""".block-container {padding-top: 1.2rem; padding-bottom: 2rem;}.hero {background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #0ea5e9 100%);padding: 1.4rem 1.5rem;border-radius: 20px;color: white;box-shadow: 0 10px 30px rgba(15, 23, 42, 0.20);margin-bottom: 1rem;}.card {background: white;border: 1px solid #e5e7eb;border-radius: 18px;padding: 1rem 1rem;box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);}.small-note {color: #64748b; font-size: 0.92rem;}.badge {display: inline-block;padding: 0.25rem 0.65rem;border-radius: 999px;background: #e0f2fe;color: #075985;font-size: 0.8rem;margin-right: 0.35rem;}""",unsafe_allow_html=True,)

RANDOM_STATE = 42

from pathlib import Path

Carpeta raíz del proyecto (donde está app.py)

BASE_DIR = Path(file).resolve().parent

Rutas correctas dentro del proyecto

DEFAULT_MODELS_DIR = BASE_DIR / "models"DEFAULT_DATA_DIR = BASE_DIR / "data"

EXPECTED_FEATURES = ["Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking","AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality","FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression","HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal","CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE","FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL","Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks","Forgetfulness",]

OPTIONAL_TARGET = "Diagnosis"IDENTIFIER_COLS = ["PatientID", "DoctorInCharge"]CATEGORICAL_COLS = ["Gender", "Ethnicity"]BINARY_COLS = ["Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes","Depression", "HeadInjury", "Hypertension", "MemoryComplaints","BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges","DifficultyCompletingTasks", "Forgetfulness",]INT_LIKE_COLS = ["Age", "EducationLevel", "SystolicBP", "DiastolicBP"] + BINARY_COLS + CATEGORICAL_COLS

MODEL_FILENAMES = {"Regresión logística": ["modelo_regresion_logistica_alzheimer.pkl",],"Random Forest - GridSearch (pocos datos)": ["modelo_random_forest_pocos_datos.pkl",],"Random Forest - RandomizedSearch (más datos)": ["modelo_random_forest_mas_datos.pkl",],"Clustering": ["cluster_kmeans_full_model.pkl",],}FEATURE_LABELS = {"Age": "Edad","Gender": "Género (0/1)","Ethnicity": "Etnia (0/1/2/3)","EducationLevel": "Nivel educativo (0/1/2/3)","BMI": "IMC","Smoking": "Tabaquismo (0/1)","AlcoholConsumption": "Consumo de alcohol","PhysicalActivity": "Actividad física","DietQuality": "Calidad de dieta","SleepQuality": "Calidad de sueño","FamilyHistoryAlzheimers": "Antecedente familiar (0/1)","CardiovascularDisease": "Enfermedad cardiovascular (0/1)","Diabetes": "Diabetes (0/1)","Depression": "Depresión (0/1)","HeadInjury": "Lesión en la cabeza (0/1)","Hypertension": "Hipertensión (0/1)","SystolicBP": "Presión sistólica","DiastolicBP": "Presión diastólica","CholesterolTotal": "Colesterol total","CholesterolLDL": "Colesterol LDL","CholesterolHDL": "Colesterol HDL","CholesterolTriglycerides": "Triglicéridos","MMSE": "MMSE","FunctionalAssessment": "Evaluación funcional","MemoryComplaints": "Quejas de memoria (0/1)","BehavioralProblems": "Problemas de conducta (0/1)","ADL": "Actividades de la vida diaria","Confusion": "Confusión (0/1)","Disorientation": "Desorientación (0/1)","PersonalityChanges": "Cambios de personalidad (0/1)","DifficultyCompletingTasks": "Dificultad para completar tareas (0/1)","Forgetfulness": "Olvidos (0/1)",}

==========================================================

Utilidades de carga y validación

==========================================================

@st.cache_data(show_spinner=False)def load_reference_dataset(data_path: str) -> Optional[pd.DataFrame]:p = Path(data_path)if p.exists() and p.suffix.lower() in [".xlsx", ".xls", ".csv"]:if p.suffix.lower() == ".csv":return pd.read_csv(p)return pd.read_excel(p)return None

def find_model_file(models_dir: str, candidates: List[str]) -> Optional[Path]:base = Path(models_dir)for fname in candidates:p = base / fnameif p.exists():return preturn None

@st.cache_resource(show_spinner=False)def load_joblib_obj(path: str):return joblib.load(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:df = df.copy()df.columns = [str(c).strip() for c in df.columns]return df

def coerce_numeric(series: pd.Series) -> pd.Series:return pd.to_numeric(series, errors="coerce")

def load_models(models_dir: str) -> Dict[str, object]:loaded = {}for model_name, candidates in MODEL_FILENAMES.items():path = find_model_file(models_dir, candidates)if path is not None:loaded[model_name] = load_joblib_obj(str(path))loaded[f"{model_name}__path"] = str(path)return loaded

def build_schema_from_reference(df_ref: Optional[pd.DataFrame]) -> Dict[str, Dict]:schema = {}if df_ref is None:# Fallback manual; sensible defaultsmins_maxs = {"Age": (60, 90), "BMI": (15.0, 40.0), "AlcoholConsumption": (0.0, 20.0),"PhysicalActivity": (0.0, 10.0), "DietQuality": (0.0, 10.0), "SleepQuality": (4.0, 10.0),"SystolicBP": (90, 180), "DiastolicBP": (60, 120), "CholesterolTotal": (150.0, 300.0),"CholesterolLDL": (50.0, 200.0), "CholesterolHDL": (20.0, 100.0),"CholesterolTriglycerides": (50.0, 400.0), "MMSE": (0.0, 30.0),"FunctionalAssessment": (0.0, 10.0), "ADL": (0.0, 10.0),}for c in EXPECTED_FEATURES:if c in CATEGORICAL_COLS:schema[c] = {"type": "categorical", "options": [0, 1] if c == "Gender" else [0, 1, 2, 3]}elif c in BINARY_COLS:schema[c] = {"type": "binary", "options": [0, 1]}elif c in INT_LIKE_COLS:schema[c] = {"type": "int", "min": mins_maxs.get(c, (0, 100))[0], "max": mins_maxs.get(c, (0, 100))[1]}else:schema[c] = {"type": "float", "min": mins_maxs.get(c, (0.0, 1.0))[0], "max": mins_maxs.get(c, (0.0, 1.0))[1]}return schema

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

def validate_and_prepare_csv(df_in: pd.DataFrame, schema: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:df = normalize_columns(df_in)issues = []

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

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if "model_results" not in st.session_state:st.session_state["model_results"] = {}

def evaluate_binary_model(y_true, y_pred, y_prob=None):metrics = {"accuracy": accuracy_score(y_true, y_pred),"precision": precision_score(y_true, y_pred, zero_division=0),"recall": recall_score(y_true, y_pred, zero_division=0),  # sensibilidad"f1": f1_score(y_true, y_pred, zero_division=0),}if y_prob is not None:try:metrics["auc"] = roc_auc_score(y_true, y_prob)except Exception:metrics["auc"] = np.nanelse:metrics["auc"] = np.nan

cm = confusion_matrix(y_true, y_pred)
return metrics, cm

def interpret_model(name, metrics, cm):tn, fp, fn, tp = cm.ravel()

parts = [
    f"**{name}**",
    f"Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Sensibilidad/Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}.",
    f"Matriz de confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}."
]

if metrics["precision"] > metrics["recall"]:
    parts.append("El modelo es más estricto al predecir positivos: comete menos falsos positivos, pero puede dejar pasar algunos casos reales.")
elif metrics["recall"] > metrics["precision"]:
    parts.append("El modelo detecta mejor los positivos reales: reduce falsos negativos, lo cual es útil cuando no conviene dejar pasar casos positivos.")
else:
    parts.append("El balance entre precisión y sensibilidad es bastante parejo.")

if fn < fp:
    parts.append("En este caso, los falsos negativos son menores que los falsos positivos, lo cual suele ser favorable en problemas médicos.")
elif fp < fn:
    parts.append("Hay más falsos negativos que falsos positivos, así que convendría revisar si el modelo está dejando pasar casos positivos.")
else:
    parts.append("Los errores positivos y negativos están equilibrados.")

return " ".join(parts)

def plot_confusion_matrix_plotly(cm, title):tn, fp, fn, tp = cm.ravel()z = [[tn, fp], [fn, tp]]

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=["Predicho 0", "Predicho 1"],
        y=["Real 0", "Real 1"],
        text=z,
        texttemplate="%{text}",
        showscale=True
    )
)
fig.update_layout(
    title=title,
    xaxis_title="Predicción",
    yaxis_title="Valor real",
    width=700,
    height=500
)
return fig

def to_model_input(df_prepared: pd.DataFrame) -> pd.DataFrame:x = df_prepared[EXPECTED_FEATURES].copy()# Mantener enteros donde aplicafor c in INT_LIKE_COLS:if c in x.columns:x[c] = x[c].round(0).astype("Int64")# Numéricos a floatfor c in x.columns:if c not in INT_LIKE_COLS:x[c] = x[c].astype(float)return x

==========================================================

Predicción y visualizaciones

==========================================================

def predict_supervised(model, x: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:preds = model.predict(x)probs = Noneif hasattr(model, "predict_proba"):probs = model.predict_proba(x)[:, 1]return preds, probs

def get_rf_importance(model, x_columns: List[str]) -> Optional[pd.DataFrame]:try:feature_names = model.named_steps["preprocess"].get_feature_names_out()importances = model.named_steps["model"].feature_importances_out = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)return outexcept Exception:return None

def get_lr_coefficients(model) -> Optional[pd.DataFrame]:try:feature_names = model.named_steps["preprocess"].get_feature_names_out()coef = model.named_steps["model"].coef_[0]out = pd.DataFrame({"feature": feature_names, "coef": coef})out["abs_coef"] = out["coef"].abs()out = out.sort_values("abs_coef", ascending=False)return outexcept Exception:return None

def predict_cluster(artifact, x: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:# Soporta: KMeans directo, dict con preprocessor + cluster_model, o pipeline completoif isinstance(artifact, dict):pre = artifact.get("preprocessor")cluster_model = artifact.get("cluster_model")if pre is not None and cluster_model is not None:x_t = pre.transform(x)labels = cluster_model.predict(x_t) if hasattr(cluster_model, "predict") else cluster_model.fit_predict(x_t)# Proyección PCA para vizpca = PCA(n_components=2, random_state=RANDOM_STATE)coords = pca.fit_transform(x_t)viz = pd.DataFrame(coords, columns=["PC1", "PC2"])viz["Cluster"] = labelsreturn labels, viz# pipeline o modelo directoif hasattr(artifact, "predict"):labels = artifact.predict(x)# intento de usar transform si existe preprocessor dentro de pipelinetry:x_t = artifact.named_steps["preprocess"].transform(x)pca = PCA(n_components=2, random_state=RANDOM_STATE)coords = pca.fit_transform(x_t)viz = pd.DataFrame(coords, columns=["PC1", "PC2"])viz["Cluster"] = labelsreturn labels, vizexcept Exception:viz = pd.DataFrame({"Cluster": labels})return labels, vizraise ValueError("No se pudo interpretar el artefacto de clustering.")

==========================================================

UI

==========================================================

ref_path_default = str(DEFAULT_DATA_DIR / "alzheimer_dataset.xlsx")models_dir_default = str(DEFAULT_MODELS_DIR)

with st.sidebar:st.markdown("## ⚙️ Configuración")models_dir = st.text_input("Carpeta de modelos", models_dir_default)reference_data_path = st.text_input("Dataset de referencia", ref_path_default)st.caption("Se usa para validar columnas, tipos y rangos.")

reference_df = load_reference_dataset(reference_data_path)schema = build_schema_from_reference(reference_df)models = load_models(models_dir)

st.markdown("""🧠 Alzheimer ML DashboardCarga CSV, captura datos manualmente, valida tipos/rangos y ejecuta predicciones con una interfaz clara y profesional.""",unsafe_allow_html=True,)

col_a, col_b, col_c = st.columns(3)with col_a:st.metric("Modelos cargados", len([k for k in models.keys() if not k.endswith("__path")]))with col_b:st.metric("Variables de entrada", len(EXPECTED_FEATURES))with col_c:st.metric("Claves del esquema", len(schema))

if reference_df is None:st.warning("No se encontró el dataset de referencia. La validación usará rangos genéricos.")else:st.success(f"Dataset de referencia cargado: {reference_df.shape[0]:,} filas × {reference_df.shape[1]:,} columnas")

Estado de sesión

if "uploaded_df" not in st.session_state:st.session_state["uploaded_df"] = Noneif "manual_df" not in st.session_state:st.session_state["manual_df"] = pd.DataFrame(columns=EXPECTED_FEATURES)if "validated_csv" not in st.session_state:st.session_state["validated_csv"] = Noneif "csv_issues" not in st.session_state:st.session_state["csv_issues"] = []

if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}

tab1, tab2, tab3, tab4 = st.tabs(["📁 Cargar CSV","✍️ Captura manual","🤖 Predicciones","📊 Hallazgos",])

==========================================================

TAB 1: CSV

==========================================================

with tab1:st.subheader("Carga de archivo CSV")uploaded = st.file_uploader("Sube un archivo CSV con las variables del modelo", type=["csv"])

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

==========================================================

TAB 2: Captura manual

==========================================================

with tab2:st.subheader("Captura manual de un registro")st.write("Completa el formulario y agrega el registro a la cola de predicción.")

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

==========================================================

TAB 3: Predicciones

==========================================================

with tab3:st.subheader("Ejecutar modelos")

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

if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}

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

            # ==================================================
            # MODELOS SUPERVISADOS
            # ==================================================
            if model_name in [
                "Regresión logística",
                "Random Forest - GridSearch (pocos datos)",
                "Random Forest - RandomizedSearch (más datos)",
            ]:
                preds, probs = predict_supervised(model_obj, x_in)

                # Decisión clínica priorizando sensibilidad.
                # Si hay probabilidad, usamos umbral ajustable.
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
                    "No tiene diagnóstico"
                )

                results_frames.append(out)

                # Métricas rápidas de lo predicho
                c1, c2, c3 = st.columns(3)
                c1.metric("Tiene diagnóstico", int(np.sum(clinical_pred == 1)))
                c2.metric("No tiene diagnóstico", int(np.sum(clinical_pred == 0)))
                if probs is not None:
                    c3.metric("Probabilidad media", f"{np.mean(probs):.3f}")
                else:
                    c3.metric("Probabilidad media", "N/A")

                st.dataframe(out.head(50), use_container_width=True)

                # Gráfico de salida clínica
                summary = pd.DataFrame({
                    "Resultado clínico": ["Tiene diagnóstico", "No tiene diagnóstico"],
                    "Cantidad": [
                        int(np.sum(clinical_pred == 1)),
                        int(np.sum(clinical_pred == 0)),
                    ]
                })

                fig_summary = px.bar(
                    summary,
                    x="Resultado clínico",
                    y="Cantidad",
                    title=f"Resumen clínico - {model_name}",
                )
                st.plotly_chart(fig_summary, use_container_width=True)

                # Histograma de probabilidades
                if probs is not None:
                    fig = px.histogram(
                        pd.DataFrame({"Probabilidad": probs}),
                        x="Probabilidad",
                        nbins=20,
                        title=f"Distribución de probabilidades - {model_name}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Importancia de variables para Random Forest
                if "Random Forest" in model_name:
                    imp = get_rf_importance(model_obj, EXPECTED_FEATURES)
                    if imp is not None:
                        fig_imp = px.bar(
                            imp.head(15).iloc[::-1],
                            x="importance",
                            y="feature",
                            orientation="h",
                            title=f"Importancia de variables - {model_name}",
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                # Coeficientes para regresión logística
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

                # Guardar resultados para la tab 4
                # Solo guardamos y_true si el origen es CSV validado
                # y existe Diagnosis en el archivo original.
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
                        y_true = pd.to_numeric(
                            uploaded_df[OPTIONAL_TARGET],
                            errors="coerce"
                        ).fillna(0).astype(int).values

                st.session_state["model_results"][model_name] = {
                    "kind": "supervised",
                    "preds": preds,
                    "probs": probs,
                    "clinical_pred": clinical_pred,
                    "threshold": sensitivity_threshold,
                    "y_true": y_true,
                }

            # ==================================================
            # CLUSTERING
            # ==================================================
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

                st.session_state["model_results"][model_name] = {
                    "kind": "cluster",
                    "labels": labels,
                }

        # ==================================================
        # COMPARACIÓN ENTRE LOS DOS RANDOM FOREST
        # ==================================================
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

with tab4:st.subheader("Hallazgos más relevantes")

st.markdown(
    """
    <div class="card">
        <span class="badge">Interpretación automática</span>
        <span class="badge">Matriz de confusión</span>
        <span class="badge">Sensibilidad priorizada</span>
        <p style="margin-top:0.8rem; margin-bottom:0;">
            Esta sección compara automáticamente los modelos ejecutados, priorizando la sensibilidad
            para decidir cuál es el mejor en un contexto clínico.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

results = st.session_state.get("model_results", {})

def compute_specificity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def build_supervised_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # sensibilidad
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = compute_specificity(cm)

    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
    }

def interpret_supervised_model(model_name, metrics):
    cm = metrics["cm"]
    tn, fp, fn, tp = cm.ravel()

    text = []
    text.append(f"**{model_name}**")
    text.append(
        f"Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
        f"Sensibilidad/Recall={metrics['recall']:.3f}, Especificidad={metrics['specificity']:.3f}, "
        f"F1={metrics['f1']:.3f}"
        + (f", AUC={metrics['auc']:.3f}." if not np.isnan(metrics["auc"]) else ".")
    )
    text.append(f"Matriz de confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}.")

    if metrics["recall"] >= metrics["precision"]:
        text.append(
            "El modelo prioriza la detección de casos positivos, lo que es favorable cuando el objetivo es no dejar pacientes con posible diagnóstico sin identificar."
        )
    else:
        text.append(
            "El modelo es más conservador al etiquetar positivos, lo que reduce falsos positivos pero puede perder algunos casos reales."
        )

    if fn < fp:
        text.append(
            "Los falsos negativos son menores que los falsos positivos, lo cual es especialmente útil en un contexto médico."
        )
    elif fn > fp:
        text.append(
            "Hay más falsos negativos que falsos positivos; convendría revisar el umbral para mejorar la sensibilidad."
        )
    else:
        text.append("Los errores de ambos tipos están equilibrados.")

    return " ".join(text)

def plot_cm_heatmap(cm, title):
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

supervised_rows = []
supervised_available = False

for model_name, info in results.items():
    if info.get("kind") != "supervised":
        continue

    supervised_available = True
    y_pred = info.get("clinical_pred", info.get("preds"))
    y_prob = info.get("probs")
    y_true = info.get("y_true")

    # Si hay diagnóstico real, calculamos métricas reales.
    # Si no hay y_true, solo damos resumen descriptivo.
    if y_true is not None and len(y_true) == len(y_pred):
        metrics = build_supervised_metrics(y_true, y_pred, y_prob)
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

if supervised_available:
    st.markdown("### Comparación de modelos supervisados")

    df_supervised = pd.DataFrame(supervised_rows)

    if df_supervised["Sensibilidad"].notna().any():
        df_supervised = df_supervised.sort_values(
            by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
            ascending=False
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
        use_container_width=True
    )

    # Gráfica comparativa
    metric_cols = [c for c in ["Precision", "Sensibilidad", "Especificidad", "F1", "Accuracy", "AUC"] if c in df_supervised.columns]
    plot_df = df_supervised.melt(id_vars="Modelo", value_vars=metric_cols, var_name="Métrica", value_name="Valor")

    fig_metrics = px.bar(
        plot_df,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Comparación de desempeño entre modelos supervisados"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Elegir mejor modelo: primero sensibilidad, luego F1
    if df_supervised["Sensibilidad"].notna().any():
        best_row = df_supervised.sort_values(
            by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
            ascending=False
        ).iloc[0]

        st.success(
            f"**Mejor modelo seleccionado: {best_row['Modelo']}**. "
            f"Se eligió porque obtuvo la mayor sensibilidad, que es el criterio principal en este proyecto. "
            f"Además, mantiene un F1 de {best_row['F1']:.3f}, lo que indica un balance razonable entre precisión y sensibilidad."
        )

        st.markdown("### Interpretación por modelo")

        for _, row in df_supervised.iterrows():
            model_name = row["Modelo"]
            info = results[model_name]
            y_pred = info.get("clinical_pred", info.get("preds"))
            y_prob = info.get("probs")
            y_true = info.get("y_true")

            st.subheader(model_name)

            if y_true is not None and len(y_true) == len(y_pred):
                metrics = build_supervised_metrics(y_true, y_pred, y_prob)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                c2.metric("Precision", f"{metrics['precision']:.3f}")
                c3.metric("Sensibilidad", f"{metrics['recall']:.3f}")
                c4.metric("Especificidad", f"{metrics['specificity']:.3f}")
                c5.metric("F1", f"{metrics['f1']:.3f}")

                st.plotly_chart(plot_cm_heatmap(metrics["cm"], f"Matriz de confusión - {model_name}"), use_container_width=True)

                if not np.isnan(metrics["auc"]):
                    roc_fig = go.Figure()
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name}"))
                    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash")))
                    roc_fig.update_layout(title=f"Curva ROC - {model_name}", xaxis_title="FPR", yaxis_title="TPR")
                    st.plotly_chart(roc_fig, use_container_width=True)

                st.write(interpret_supervised_model(model_name, metrics))
            else:
                st.info(
                    "No existe columna real de diagnóstico para este conjunto de datos, así que solo se muestra el resultado predictivo y la distribución de casos."
                )

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
        st.warning("No hay métricas reales disponibles para comparar porque no existe la columna Diagnosis en los datos evaluados.")

else:
    st.info("No se han ejecutado modelos supervisados todavía.")

# ==========================================================
# CLUSTERING
# ==========================================================
cluster_rows = []
for model_name, info in results.items():
    if info.get("kind") == "cluster":
        labels = info.get("labels")
        counts = pd.Series(labels).value_counts().sort_index()

        cluster_rows.append({
            "Modelo": model_name,
            "Clusters": int(len(counts)),
            "Tamaño promedio": float(counts.mean()),
            "Cluster más frecuente": int(counts.idxmax()),
        })

        st.markdown(f"### {model_name}")
        cluster_df = pd.DataFrame({
            "Cluster": counts.index.astype(str),
            "Pacientes": counts.values
        })

        fig_pie = px.pie(
            cluster_df,
            names="Cluster",
            values="Pacientes",
            title="Distribución de clusters"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(cluster_df, use_container_width=True)

        st.info(
            "El clustering no predice diagnóstico; agrupa pacientes con características similares. "
            "Esto permite explorar perfiles y patrones útiles para el socio formador."
        )

if len(cluster_rows) > 0:
    st.markdown("### Resumen de clustering")
    st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

# ==========================================================
# CONCLUSIÓN EJECUTIVA
# ==========================================================
st.markdown("### Conclusión ejecutiva")

if supervised_available and len(supervised_rows) > 0:
    df_exec = pd.DataFrame(supervised_rows)

    if df_exec["Sensibilidad"].notna().any():
        best_exec = df_exec.sort_values(
            by=["Sensibilidad", "F1", "Especificidad", "Accuracy"],
            ascending=False
        ).iloc[0]

        st.success(
            f"""

En términos clínicos, el modelo más conveniente es {best_exec['Modelo']} porque prioriza la sensibilidad,que es el criterio más importante cuando se busca no dejar casos positivos sin detectar.

Su desempeño general también es sólido en F1 y especificidad, por lo que mantiene un equilibrio aceptable entredetectar correctamente pacientes con posible diagnóstico y evitar errores excesivos.""")else:st.warning("No fue posible calcular una comparación clínica real porque los datos cargados no contienen diagnóstico verdadero.")
