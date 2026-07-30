# Automobile Price Prediction

Sistema de prediccion de precio de automoviles con entrenamiento reproducible, configuracion por YAML y trazabilidad con MLflow.

## Objetivo

- Entrenar un modelo de regresion para predecir `Selling_Price`.
- Comparar arquitecturas neuronales de manera sistematica.
- Registrar ejecuciones, artefactos y modelos en MLflow.
- Mantener pruebas automatizadas y una base clara de MLOps.

## Inicio rapido

```bash
uv sync --all-extras
uv run pytest
uv run automobile-train
```

Para revisar las ejecuciones locales:

```bash
mlflow ui
```

## Documentacion

- [Guia de tests y validaciones](tests/README.md)
- [Guia de MLflow](docs/mlflow.md)
- [Reporte del modelo y resultados](src/model_report.md)

## Configuracion principal

- [Configuracion de entrenamiento](config/train.yaml)
- [Workflow de CI](.github/workflows/tests.yml)
- [Proyecto Python](pyproject.toml)

## Resumen del flujo

1. `src/config.py` carga la configuracion.
2. `src/train.py` ejecuta el pipeline de entrenamiento.
3. `src/rnn_price_predict.py` concentra limpieza, preprocesamiento, metricas y utilidades de inferencia.
4. MLflow guarda parametros, metricas, artefactos y el modelo empaquetado.
5. Los tests validan configuracion, reglas minimas de datos, entrenamiento e inferencia.

## Estructura breve

```text
.
|-- README.md
|-- config/
|-- docs/
|   `-- mlflow.md
|-- src/
|   `-- model_report.md
|-- tests/
|   `-- README.md
`-- pyproject.toml
```
