# 🧠 Alzheimer ML Dashboard

Aplicación de Streamlit para predicción de Alzheimer usando múltiples modelos de Machine Learning.

## 📋 Estructura del Proyecto

```
reto-alz/
├── app.py                    # Aplicación principal de Streamlit
├── requirements.txt          # Dependencias de Python
├── README.md                 # Este archivo
├── models/                   # Carpeta para guardar modelos entrenados
│   ├── modelo_regresion_logistica_alzheimer.pkl
│   ├── modelo_random_forest_pocos_datos.pkl
│   ├── modelo_random_forest_mas_datos.pkl
│   └── cluster_kmeans_full_model.pkl
└── data/                     # Carpeta para datos de referencia
    └── alzheimer_dataset.xlsx
```

## 🚀 Instalación y Ejecución

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar o descargar el repositorio:**
   ```bash
   git clone https://github.com/diecass/reto-alz.git
   cd reto-alz
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   # En Windows
   python -m venv venv
   venv\Scripts\activate

   # En macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
   ```

5. **Acceder a la aplicación:**
   - La aplicación se abrirá automáticamente en tu navegador
   - Si no, ingresa a `http://localhost:8501`

## 📁 Preparación de archivos

### Modelos entrenados
Coloca los archivos `.pkl` de tus modelos en la carpeta `models/`:
- `modelo_regresion_logistica_alzheimer.pkl`
- `modelo_random_forest_pocos_datos.pkl`
- `modelo_random_forest_mas_datos.pkl`
- `cluster_kmeans_full_model.pkl`

### Dataset de referencia
Coloca tu archivo `alzheimer_dataset.xlsx` en la carpeta `data/`. Este archivo se utiliza para:
- Validar columnas y tipos de datos
- Establecer rangos de valores
- Proporcionar valores por defecto en los formularios

### CSV de entrada
Puedes cargar archivos CSV con las siguientes columnas esperadas:
- Age, Gender, Ethnicity, EducationLevel, BMI, Smoking
- AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality
- FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression
- HeadInjury, Hypertension, SystolicBP, DiastolicBP
- CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides
- MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL
- Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness
- (Opcional) Diagnosis: para validar el desempeño de los modelos

## 🎯 Características principales

### Pestaña 1: Carga de CSV
- Carga y valida archivos CSV
- Verifica tipos de datos y rangos
- Muestra estadísticas de validación

### Pestaña 2: Captura manual
- Formulario interactivo para ingresar datos de un paciente
- Validación en tiempo real según el esquema
- Almacenamiento de registros para predicción posterior

### Pestaña 3: Predicciones
- Ejecuta múltiples modelos simultáneamente
- Ajusta el umbral de sensibilidad
- Visualiza distribuciones de probabilidades
- Comparación entre modelos Random Forest
- Descargas de resultados en CSV

### Pestaña 4: Hallazgos
- Comparación automática de desempeño
- Matrices de confusión interactivas
- Curvas ROC
- Interpretación clínica de resultados
- Análisis de clustering
- Conclusión ejecutiva

## 🔧 Configuración

En la barra lateral puedes configurar:
- **Carpeta de modelos**: Ruta donde se encuentran los modelos `.pkl`
- **Dataset de referencia**: Ruta del archivo Excel con datos de referencia

## 📊 Modelos soportados

1. **Regresión Logística**: Modelo de clasificación simple y interpretable
2. **Random Forest - GridSearch**: Para conjuntos pequeños de datos
3. **Random Forest - RandomizedSearch**: Para conjuntos más grandes
4. **Clustering (KMeans)**: Agrupación no supervisada de pacientes

## 📈 Métricas calculadas

- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de predicciones positivas correctas
- **Sensitivity/Recall**: Capacidad de detectar casos positivos reales
- **Specificity**: Capacidad de detectar casos negativos reales
- **F1 Score**: Balance entre precisión y sensibilidad
- **AUC-ROC**: Área bajo la curva de características operativas

## 🐛 Troubleshooting

### "Faltan modelos por cargar"
- Asegúrate de que los archivos `.pkl` existen en la carpeta `models/`
- Verifica que los nombres de archivo coinciden exactamente

### "No se encontró el dataset de referencia"
- Coloca el archivo `alzheimer_dataset.xlsx` en la carpeta `data/`
- O proporciona la ruta correcta en la configuración de la barra lateral

### Error al cargar el CSV
- Verifica que el archivo tiene la extensión `.csv`
- Asegúrate de que contiene las columnas esperadas
- Revisa el formato de los datos (números, no texto)

## 📝 Licencia

Este proyecto es parte del reto Alzheimer.

## 👤 Autor

**Diego Casseres**
- GitHub: [@diecass](https://github.com/diecass)

## 📞 Soporte

Para reportar problemas o sugerencias, abre un issue en el repositorio.
