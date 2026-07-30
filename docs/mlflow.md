# Guia de MLflow

Este documento explica como usa MLflow este proyecto, que se guarda, donde se guarda y por que se hace asi.

## Rol de MLflow en el proyecto

MLflow se usa para:

- registrar parametros de ejecucion;
- registrar metricas;
- guardar artefactos;
- versionar el modelo final;
- facilitar la comparacion entre experimentos;
- habilitar el registro en Model Registry.

La idea no es solo guardar un modelo, sino conservar trazabilidad suficiente para reproducir y auditar la corrida.

## Configuracion de tracking

La resolucion de tracking sigue este orden:

1. `tracking_uri` explicito, si existe.
2. `tracking_dir` local, convertido a `file:///...`.
3. backend local por defecto de MLflow, si no se definio nada.

### Campos de configuracion

- `tracking_uri`: servidor remoto o backend personalizado.
- `tracking_dir`: carpeta local para el backend file-based.
- `experiment_name`: nombre del experimento principal.
- `run_name`: nombre opcional de la ejecucion principal.

### Resultado esperado

- si se usa `tracking_dir: mlruns`, MLflow guarda ejecuciones en esa carpeta local;
- si se usa un `tracking_uri` remoto, las ejecuciones se envian a ese servidor;
- el codigo no fija esa decision en una constante.

## Estructura de ejecuciones

El pipeline genera:

- una ejecucion principal para toda la corrida;
- nested runs para cada experimento;
- artefactos por experimento;
- un resumen consolidado final.

Esto permite leer el proceso en dos niveles:

- nivel global: ejecucion completa;
- nivel local: comparacion de candidatos.

## Que se registra

### Tags

Tags de trazabilidad:

- `git_commit`
- `config_path`
- `data_path`
- `data_rows`
- `feature_columns`
- `status`
- `registry_status`
- metadatos de rutas MLflow: `mlflow_tracking_uri`, `mlflow_tracking_dir`, `mlflow_model_artifact_path`, `mlflow_config_artifact_path`, `mlflow_history_artifact_path`, `mlflow_summary_artifact_path`, `mlflow_reports_artifact_dir`

Motivo:

- los tags ayudan a buscar y filtrar ejecuciones;
- no mezclan datos con configuracion del modelo.

### Parametros

Se registran, entre otros:

- target;
- split;
- epocas;
- batch size;
- patience;
- learning rate;
- criterio de seleccion;
- arquitectura ganadora;
- URI del modelo.

Motivo:

- los parametros documentan la decision de entrenamiento;
- permiten comparar ejecuciones con distinta configuracion.

### Metricas

Se registran:

- baseline;
- metricas de validacion;
- metricas de test;
- metrica de seleccion del mejor candidato.

Motivo:

- sirven para elegir el modelo ganador;
- permiten evaluar si hubo mejora real sobre una referencia simple.

## Artefactos y rutas

Rutas parametrizadas actuales:

- `tracking_dir`: `mlruns`
- `model_artifact_path`: `model`
- `config_artifact_path`: `config/config.json`
- `history_artifact_path`: `reports/final_training_history.json`
- `summary_artifact_path`: `reports/experiment_summary.json`
- `reports_artifact_dir`: `reports`

### Que guarda cada uno

- `config/config.json`: snapshot de configuracion resuelta.
- `reports/final_training_history.json`: historia del mejor candidato.
- `reports/experiment_summary.json`: resumen JSON de todos los experimentos.
- `reports/`: carpeta para reportes CSV.
- `model/`: modelo `pyfunc` final.

### Preprocesador y modelo

El modelo final se guarda como un `pyfunc` que empaqueta:

- preprocesador serializado;
- modelo Keras serializado;
- logica de limpieza e inferencia.

Motivo:

- el consumidor no debe reconstruir la pipeline a mano;
- el artefacto debe aceptar el `DataFrame` crudo del dominio.

## Model Registry

Si `registered_model_name` existe y no se usa `--no-registration`, el modelo ganador se registra en MLflow Model Registry.

El flujo es:

1. entrenar y evaluar;
2. identificar el ganador global;
3. registrar la URI del modelo;
4. intentar el registro;
5. si falla, guardar el error en un tag sin abortar la corrida.

## Como inspeccionar resultados

### UI local

```bash
mlflow ui
```

### Artefactos esperados por ejecucion

- parametros y metricas en la ejecucion principal y en los nested runs;
- modelo bajo `model/`;
- archivos de reporte bajo `reports/`;
- snapshot de config bajo `config/`.

## Por que esta documentacion existe

Porque MLflow mezcla varias capas:

- experimentos;
- ejecuciones;
- tags;
- parametros;
- metricas;
- artefactos;
- registry.

Sin una guia especifica, es facil perder la relacion entre lo que se entrena, lo que se guarda y lo que luego se consume.
