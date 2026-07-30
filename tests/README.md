# Tests y validaciones

Este documento explica que valida cada prueba, por que existe y que parte del proyecto protege.

## Objetivo del conjunto de tests

Las pruebas buscan asegurar que:

- la configuracion tenga valores por defecto coherentes;
- el dataset cumpla un conjunto minimo de reglas;
- la limpieza y el preprocesamiento no cambien de forma silenciosa;
- el modelo `pyfunc` reciba entradas crudas y devuelva predicciones consistentes;
- el pipeline de entrenamiento siga produciendo ejecuciones de MLflow validas;
- la seleccion del mejor modelo sea reproducible.

## Mapa de pruebas

| Archivo | Que valida | Por que existe |
| --- | --- | --- |
| `tests/test_config.py` | Construccion de `DataConfig`, `TrainingConfig` y `ExperimentConfig` desde YAML o mappings vacios. | Evita que un cambio en los valores por defecto rompa el arranque del proyecto o cambie la interpretacion de la configuracion. |
| `tests/test_data_contract.py` | Reglas minimas del dataset, existencia del archivo y columnas necesarias. | Protege el pipeline contra datasets incompletos o rutas incorrectas. |
| `tests/test_utils.py` | Limpieza, metricas, split, preprocesador y comparacion de arquitecturas. | Evita regresiones en la logica central de entrenamiento. |
| `tests/test_pyfunc.py` | Reglas de inferencia del wrapper MLflow `pyfunc`. | Garantiza que el modelo registrado pueda consumirse con un `DataFrame` crudo. |
| `tests/test_train.py` | Ejecucion end-to-end del pipeline con MLflow simulado. | Asegura que el orquestador siga ensamblando el flujo completo sin reentrenar el modelo seleccionado. |

## Validaciones y su proposito

### Configuracion

Valida que:

- `config/train.yaml` se interprete correctamente;
- los valores por defecto existan aunque el YAML venga vacio;
- las rutas y los parametros de entrenamiento no queden desalineados.

Motivo:

- el proyecto depende mucho de configuracion declarativa;
- una configuracion mal interpretada cambia silenciosamente el experimento.

### Dataset

Valida que:

- el archivo exista;
- la columna objetivo exista;
- haya al menos columnas categoricas o numericas utilizables;
- el split deje datos suficientes para entrenar, validar y evaluar.

Motivo:

- el pipeline no puede asumir que el CSV siempre esta completo;
- si fallan las reglas del dato, el resto del entrenamiento deja de ser confiable.

### Limpieza y preprocesamiento

Valida que:

- los tokens faltantes se normalicen;
- los valores vacios no rompan el pipeline;
- las columnas categoricas y numericas no se mezclen;
- el preprocesador acepte categorias no vistas;
- la transformacion produzca dimensiones consistentes.

Motivo:

- la red neuronal solo funciona si la entrada queda estable y numerica;
- la limpieza debe ser predecible entre entrenamiento e inferencia.

### Metricas y seleccion

Valida que:

- MAE, MSE, RMSE, MAPE y R2 se calculen correctamente;
- la comparacion de arquitecturas ordene por el criterio correcto;
- el mejor candidato quede identificado de forma determinista.

Motivo:

- el proyecto compara modelos entre si;
- si la metrica o el orden cambian, tambien cambia el ganador.

### Modelo `pyfunc`

Valida que:

- la entrada sea un `DataFrame`;
- si faltan columnas, el error sea claro;
- el wrapper cargue artefactos serializados;
- la salida tenga formato tabular.

Motivo:

- MLflow debe guardar un artefacto portable y consumible;
- el consumidor no debe reconstruir la pipeline a mano.

### Entrenamiento e integracion

Valida que:

- el pipeline registre parametros, tags y artefactos;
- el modelo seleccionado no se reentrene al final;
- el run padre y los nested runs se creen correctamente;
- el resumen final del experimento exista.

Motivo:

- protege el comportamiento end-to-end real del proyecto;
- evita que un refactor rompa MLflow, la seleccion o el empaquetado.

## Como ejecutar

### Suite completa

```bash
uv run pytest
```

### Solo tests unitarios

```bash
uv run pytest -m "not integration"
```

### Solo integracion

```bash
uv run pytest -m integration
```

## Que revisar cuando falla un test

1. Si falla `test_config`, revisar `config/train.yaml` y `src/config.py`.
2. Si falla `test_utils`, revisar limpieza, metricas o split en `src/rnn_price_predict.py`.
3. Si falla `test_pyfunc`, revisar el wrapper de inferencia y la serializacion de artefactos.
4. Si falla `test_train`, revisar `src/train.py` y las rutas de MLflow.

## Relacion con CI

El workflow de GitHub Actions se divide en dos partes:

- `push` sobre `main`: ejecuta los tests unitarios con `pytest -m "not integration"`.
- `pull_request` y `workflow_dispatch`: ejecutan los tests unitarios con `pytest -m "not integration"`.
- `schedule` y `workflow_dispatch`: ejecutan los tests de integracion con `pytest -m integration`.

Esto permite:

- validar rapido los cambios comunes en cada envio;
- reservar los tests mas pesados para ejecuciones manuales o programadas;
- mantener alineado el pipeline con la version de Python que declara el proyecto.
