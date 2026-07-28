# Automobile Price Prediction

Sistema de prediccion de precio de automoviles con un flujo de entrenamiento reproducible, configuracion por YAML, trazabilidad con MLflow y una base inicial de MLOps.

El proyecto fue refactorizado para separar la logica reutilizable del orquestador de entrenamiento, facilitar experimentacion, y dejar un camino claro para versionado de modelos, pruebas automaticas y despliegue futuro.

## Objetivos del proyecto

- Entrenar un modelo de regresion para predecir `Selling_Price`.
- Reutilizar funciones de limpieza, preprocesamiento, evaluacion y construccion de modelo.
- Comparar varias arquitecturas de red neuronal en una sola corrida.
- Registrar parametros, metricas, artefactos y modelos en MLflow.
- Mantener el proyecto configurado con `pyproject.toml` y ejecutable con `uv`.
- Dejar una base razonable para buenas practicas de MLOps.

## Estado actual

El flujo actual ya incluye:

- lectura de configuracion desde `config/train.yaml`
- carga del dataset desde una ruta configurable
- limpieza de columnas categoricas y numericas
- preprocesamiento con `ColumnTransformer`
- comparacion de arquitecturas Keras
- seleccion del mejor experimento por `rmse_valid`
- entrenamiento final
- logging de MLflow
- registro opcional en MLflow Model Registry
- pruebas automatizadas con `pytest`

## Estructura del repositorio

```text
.
|-- config/
|   `-- train.yaml
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- train.py
|   `-- utils.py
|-- tests/
|   |-- test_config.py
|   `-- test_utils.py
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|-- pyproject.toml
`-- README.md
```

### Archivos principales

- `src/utils.py`
  - funciones reutilizables de limpieza, preprocessing, metricas, split y helpers de MLflow
- `src/config.py`
  - dataclasses de configuracion y carga de YAML
- `src/train.py`
  - orquestacion completa del pipeline de entrenamiento y logging
- `config/train.yaml`
  - configuracion base del proyecto y suite de experimentos
- `tests/`
  - pruebas unitarias para config y utilidades
- `pyproject.toml`
  - configuracion del proyecto, dependencias y entrypoint CLI

## Requisitos

- Python 3.10 o superior
- `uv`
- Acceso al dataset de automoviles en la ruta configurada
- Dependencias de ML instalables en el entorno local

## Instalacion

La forma oficial de trabajar con este proyecto es `uv` + `pyproject.toml`.

### Sincronizar dependencias

```bash
uv sync --all-extras
```

Esto instala:

- dependencias de runtime
- dependencias de desarrollo
- el paquete local en modo editable

### Sincronizar solo runtime

```bash
uv sync
```

### Nota para Windows

En algunos entornos Windows, `uv` puede fallar si no puede usar su cache por defecto. En ese caso, usa una cache local dentro del workspace:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD ".uv-cache"
uv sync --all-extras
```

## Dependencias del proyecto

Las dependencias principales viven en `pyproject.toml`:

- `mlflow`
- `numpy`
- `pandas`
- `pyyaml`
- `scikit-learn`
- `tensorflow`
- `pytest` como extra de desarrollo

`requirements.txt` fue eliminado a proposito para evitar dos fuentes de verdad.

## Como ejecutar el entrenamiento

El entrypoint oficial del proyecto es el comando registrado en `pyproject.toml`:

```bash
uv run automobile-train
```

Ese comando:

1. lee `config/train.yaml`
2. carga el dataset
3. prepara el split train/valid/test
4. limpia y transforma las variables
5. compara arquitecturas
6. entrena el mejor candidato
7. evalua en test
8. registra el run en MLflow

## Configuracion

La configuracion por defecto esta en [`config/train.yaml`](config/train.yaml).

El archivo esta dividido en dos bloques:

- `data`
- `training`

### Bloque `data`

Define como se interpreta el dataset.

Campos principales:

- `data_path`: ruta al CSV del dataset
- `target_column`: nombre de la columna objetivo
- `categorical_columns`: columnas tratadas como categoricas
- `numeric_columns`: columnas tratadas como numericas
- `test_size`: fraccion para test
- `validation_size`: fraccion para validacion
- `random_state`: semilla de reproducibilidad

### Bloque `training`

Define la estrategia de entrenamiento y experimentacion.

Campos principales:

- `experiment_name`: nombre del experimento en MLflow
- `registered_model_name`: nombre del modelo en MLflow Registry
- `tracking_uri`: URI del tracking server de MLflow
- `epochs`: epocas maximas
- `batch_size`: tamano de batch
- `patience`: paciencia de early stopping
- `learning_rate`: tasa de aprendizaje
- `use_early_stopping`: activa o desactiva early stopping
- `verbose_training`: nivel de verbosidad de Keras
- `run_name`: nombre opcional del run
- `experiments`: lista de experimentos a comparar

## Suite de experimentos

Este proyecto ya no usa una sola corrida rigida. En su lugar, `training.experiments` permite definir varios candidatos.

Ejemplo:

```yaml
training:
  experiments:
    - name: compact-baseline
      learning_rate: 0.001
      epochs: 250
      patience: 20
      architectures:
        - [16]
        - [32]

    - name: balanced-default
      learning_rate: 0.001
      epochs: 300
      patience: 30
      architectures:
        - [64, 32]
        - [128, 64]
```

Cada experimento:

- hereda valores base si no define un campo
- compara sus arquitecturas internas
- produce su propio nested run en MLflow
- se evalua por `criterion_order`

### Criterio de seleccion

Por defecto el proyecto ordena por `rmse_valid`.

Eso significa que:

- menor `rmse_valid` es mejor
- el mejor candidato de cada experimento se selecciona primero
- luego se elige el mejor experimento global

## Como cambiar la configuracion

### Opcion 1: editar el YAML

Es la ruta recomendada para ajustes persistentes.

### Opcion 2: sobreescribir por CLI

Puedes cambiar valores puntuales al ejecutar:

```bash
uv run automobile-train --epochs 100 --batch-size 32
```

Tambien puedes cambiar ruta de datos o tracking server:

```bash
uv run automobile-train --data-path data/automobile_dataset.csv --tracking-uri http://localhost:5000
```

## MLflow

El pipeline registra en MLflow:

- parametros de ejecucion
- configuracion completa del run
- metadatos del dataset
- baseline de comparacion
- tabla resumen de arquitecturas
- historial de entrenamiento
- metricas de validacion y test
- modelo final
- ruta del modelo registrado

### Artefactos registrados

Entre los artefactos principales se guardan:

- `config/config.json`
- `reports/final_training_history.json`
- tabla CSV con comparacion de experimentos
- resumen de experimentos
- modelo serializado
- preprocesador serializado dentro del modelo pyfunc

### Modelo en MLflow

El modelo se loggea como un `pyfunc` wrapper para que pueda recibir un `DataFrame` crudo con columnas originales.

Eso permite:

- inferencia con el mismo esquema de features usado en entrenamiento
- aplicacion consistente de reglas de limpieza
- encapsular preprocessing y red neuronal en un mismo artefacto

### Model Registry

Si `registered_model_name` esta configurado y `--no-registration` no se usa, el mejor modelo puede registrarse en MLflow Model Registry.

La intencion del flujo es:

- experimentar
- comparar
- elegir un ganador
- registrar la version candidata
- promoverla luego si cumple criterios de negocio o calidad

## Ejecutar MLflow UI

Si tienes un backend local de MLflow o quieres ver los runs generados localmente:

```bash
mlflow ui
```

Luego abre el navegador en la URL que te muestre MLflow, normalmente `http://127.0.0.1:5000`.

## Pruebas automatizadas

El proyecto usa `pytest`.

### Ejecutar pruebas

```bash
uv run pytest
```

### Que validan las pruebas actuales

- configuracion de datos
- normalizacion de arquitecturas
- creacion de suite de experimentos
- logica de limpieza
- resolucion de columnas disponibles
- split train/valid/test

### CI

Hay un workflow de GitHub Actions en:

- [`.github/workflows/tests.yml`](.github/workflows/tests.yml)

Ese workflow:

- instala dependencias con `uv`
- ejecuta `pytest`
- valida el repo en cada `push` y `pull_request`

## Recomendacion de flujo de trabajo

Una rutina razonable para este proyecto es:

1. editar `config/train.yaml`
2. ejecutar `uv sync --all-extras`
3. correr `uv run pytest`
4. lanzar `uv run automobile-train`
5. revisar el experimento en MLflow
6. ajustar arquitecturas, learning rate o paciencia
7. volver a correr

## Buenas practicas MLOps ya incorporadas

### Reproducibilidad

- semilla fija
- configuracion versionada en YAML
- parametros loggeados en MLflow

### Separacion de responsabilidades

- utilidades reutilizables separadas del orquestador
- configuracion aislada del codigo de entrenamiento
- pruebas separadas del flujo de produccion

### Trazabilidad

- tracking de runs en MLflow
- logging de artefactos
- logging de tags utiles como `git_commit` y `config_path`

### Comparacion sistematica

- varias arquitecturas
- metricas comparables
- seleccion automatica del mejor candidato

### Preparacion para despliegue

- modelo encapsulado para inferencia
- `pyfunc` listo para consumo desde MLflow
- naming consistente para registry

## Limitaciones actuales

Este proyecto aun no incluye:

- API de serving en FastAPI o Flask
- pipeline de CI/CD completo para despliegue automatico
- validacion formal de esquema con librerias de data contracts
- monitoreo de drift en produccion
- versionado de dataset en un data lake o feature store

Eso no impide entrenar, comparar y registrar modelos, pero si marca el siguiente paso natural de evolucion.

## Problemas comunes

### `uv` no puede usar su cache por defecto

Usa una cache local:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD ".uv-cache"
uv sync --all-extras
```

### `uv sync` intenta bajar paquetes y falla por red restringida

En una red sin acceso a PyPI, `uv sync` no podra resolver dependencias nuevas. En ese caso:

- usa un entorno ya preparado
- o instala dependencias en una maquina con red
- o usa `uv` con un cache/repositorio interno de paquetes

### No encuentra el dataset

Verifica la ruta en `config/train.yaml` en:

```yaml
data:
  data_path: data/automobile_dataset.csv
```

Si tu CSV esta en otra ruta, actualizala o pasa `--data-path`.

## Desarrollo local

Si vas a modificar el codigo:

1. cambia primero `src/utils.py` si tocas logica reutilizable
2. deja `src/train.py` como orquestador
3. agrega o ajusta pruebas en `tests/`
4. ejecuta `uv run pytest`

## Proximos pasos recomendados

1. Crear una API de inferencia que cargue el modelo registrado desde MLflow.
2. Agregar validacion formal de schema antes de entrenar e inferir.
3. Introducir monitoreo de calidad y drift cuando el modelo vaya a produccion.
4. Añadir versionado de datos y ejecucion automatica por CI/CD.

