# Pipeline de ML con MLflow: benchmark de 7 modelos de regresión

Problema de **regresion multivariada**: entrenar un modelo con 800 muestras (20 features → 1 target continuo), predecir 200 muestras de test ciego (blind test) y entregar una solucion lista para produccion.

## Inicio Rapido

```bash
# Instalar dependencias (requiere uv y Python 3.14+)
uv sync

# Ejecutar el pipeline completo (entrenar → evaluar → predecir → guardar)
uv run python main.py
```

## Resumen del Pipeline

`main.py` es el punto de entrada y orquesta todo el flujo de trabajo en 7 pasos:

1. **Cargar datos** — lee los CSVs de entrenamiento (800 muestras) y test ciego (200 muestras) desde `data/`
2. **Dividir datos** — separa en 80% entrenamiento / 20% validacion con `train_test_split` (semilla fija `random_state=42` para reproducibilidad)
3. **Entrenar 7 modelos** — cada modelo se envuelve en un `sklearn.Pipeline` que incluye `StandardScaler` + estimador. Se optimizan hiperparametros con `RandomizedSearchCV` (50 iteraciones, validacion cruzada de 5 folds)
4. **Comparar modelos** — genera una tabla ranking ordenada por CV MAE (error absoluto medio en validacion cruzada)
5. **Evaluar en hold-out** — calcula metricas del mejor modelo sobre el conjunto de validacion reservado (MAE, RMSE, R²)
6. **Predecir test ciego** — genera predicciones para las 200 muestras → `outputs/target_pred.csv`
7. **Guardar artefacto** — serializa el pipeline completo con joblib → `models/best_model_<nombre>.joblib`

Todos los experimentos se registran con **MLflow** (tracking local en `mlruns/`).

## Modelos Entrenados

Se evaluan 7 algoritmos de regresion con busqueda aleatoria de hiperparametros:

| Modelo | Libreria | Descripcion |
|---|---|---|
| Ridge | scikit-learn | Regresion lineal con regularizacion L2 |
| ElasticNet | scikit-learn | Regresion lineal con regularizacion L1 + L2 |
| Random Forest | scikit-learn | Ensamble de arboles de decision (bagging) |
| Gradient Boosting | scikit-learn | Ensamble secuencial de arboles (boosting) |
| XGBoost | xgboost | Gradient boosting optimizado con regularizacion |
| LightGBM | lightgbm | Gradient boosting eficiente basado en histogramas |
| SVR | scikit-learn | Maquinas de soporte vectorial para regresion |

## Estructura del Proyecto

```
main.py                 # Punto de entrada — ejecuta el pipeline completo
src/
  config.py             # Configuracion: rutas, hiperparametros, ajustes de MLflow
  data_loader.py        # Carga de CSVs y division train/validacion
  models.py             # Registro de modelos (estimadores + distribuciones de hiperparametros)
  train.py              # Creacion del Pipeline, RandomizedSearchCV, logging en MLflow
  evaluate.py           # Comparacion de modelos y evaluacion en hold-out
  predict.py            # Generacion de predicciones y guardado de artefactos
data/
  training_data.csv     # 800 filas x 21 columnas (features + target)
  blind_test_data.csv   # 200 filas x 20 columnas (solo features)
models/                 # Pipeline serializado con sklearn + joblib (.joblib)
outputs/                # target_pred.csv (200 filas, columna target_pred)
notebooks/
  data_analisys.ipynb   # Analisis exploratorio de datos (EDA)
mlruns/                 # Tracking de experimentos con MLflow
```

## Descripcion de Cada Modulo

### `main.py` — Orquestador

Punto de entrada que coordina los 7 pasos del pipeline. Importa funciones de los modulos en `src/` y las ejecuta secuencialmente: carga de datos → division → entrenamiento → comparacion → evaluacion → prediccion → guardado. Imprime un resumen final con las metricas del mejor modelo.

### `src/config.py` — Configuracion Centralizada

Define todas las constantes del proyecto en un solo lugar:
- **Rutas**: directorios de datos, modelos y salidas
- **Configuracion ML**: semilla aleatoria (42), porcentaje de test (20%), folds de CV (5), iteraciones de busqueda (50)
- **MLflow**: nombre del experimento y URI de tracking local

### `src/data_loader.py` — Carga y Division de Datos

Tres funciones simples:
- `load_training_data()` — lee el CSV de entrenamiento, separa features (X) del target (y)
- `load_blind_test_data()` — lee el CSV de test ciego (solo features, sin target)
- `split_data()` — divide X e y en conjuntos de entrenamiento y validacion (80/20)

### `src/models.py` — Registro de Modelos

Define los 7 modelos como una lista de diccionarios, cada uno con:
- `name`: nombre identificador del modelo
- `estimator`: instancia del estimador de scikit-learn/xgboost/lightgbm
- `params`: distribuciones de hiperparametros para la busqueda aleatoria (prefijo `model__` para compatibilidad con Pipeline)

### `src/train.py` — Entrenamiento con Pipeline

Logica principal de entrenamiento:
- `train_single_model()` — crea un `Pipeline(StandardScaler + estimador)`, ejecuta `RandomizedSearchCV` para encontrar los mejores hiperparametros, calcula metricas de CV adicionales (MAE, R², RMSE) y registra todo en MLflow
- `train_all_models()` — itera sobre el registro de modelos y entrena cada uno

### `src/evaluate.py` — Evaluacion y Comparacion

- `compare_models()` — construye una tabla comparativa con metricas de CV de todos los modelos
- `select_best_model()` — selecciona el modelo con menor CV MAE
- `evaluate_on_holdout()` — evalua el mejor modelo en el conjunto de validacion reservado
- `print_comparison_table()` — imprime la tabla de comparacion formateada

### `src/predict.py` — Prediccion y Guardado

- `generate_predictions()` — ejecuta `pipeline.predict(X)` sobre datos crudos (el Pipeline escala internamente)
- `save_predictions()` — guarda las predicciones como CSV con columna `target_pred`
- `save_model_artifact()` — serializa el Pipeline completo con joblib

## Arquitectura

Cada modelo se envuelve en un **sklearn.Pipeline**:

```python
Pipeline([
    ("scaler", StandardScaler()),   # Paso 1: estandariza features (media=0, std=1)
    ("model",  <estimador>),        # Paso 2: modelo de regresion
])
```

Esto garantiza que el preprocesamiento y la prediccion sean una **unidad atomica**. El artefacto serializado es el Pipeline completo — para inferencia solo se necesita:

```python
import joblib

# Cargar el pipeline entrenado
pipeline = joblib.load("models/best_model_gradientboosting.joblib")

# Predecir directamente sobre datos crudos (sin escalar manualmente)
predicciones = pipeline.predict(X_crudo)
```

## Configuracion

Parametros clave definidos en `src/config.py`:

| Parametro | Valor | Descripcion |
|---|---|---|
| `RANDOM_STATE` | 42 | Semilla para reproducibilidad |
| `TEST_SIZE` | 0.2 | 20% de datos para validacion |
| `CV_FOLDS` | 5 | Folds de validacion cruzada |
| `N_ITER` | 50 | Combinaciones de hiperparametros a probar |
| `CV_SCORING` | neg_mean_absolute_error | Metrica de optimizacion |

## Stack Tecnologico

- **Python 3.14** (gestionado con `uv`)
- **scikit-learn, xgboost, lightgbm** — modelado y pipelines
- **pandas, numpy, scipy** — manipulacion de datos y distribuciones estadisticas
- **MLflow** — tracking de experimentos
- **joblib** — serializacion de modelos
- **matplotlib, seaborn** — visualizacion (notebook de EDA)
