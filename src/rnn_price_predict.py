from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class ArchitectureResult:
    architecture: tuple[int, ...]
    history: dict[str, list[float]]
    train_metrics: dict[str, float]
    valid_metrics: dict[str, float]
    epochs_trained: int
    best_epoch: int
    model: Any


@dataclass
class BestArchitectureResult:
    ranking: int
    architecture: tuple[int, ...]
    history: dict[str, list[float]]
    model: Any
    rmse_valid: float
    mae_valid: float
    r2_valid: float
    val_loss_minima: float
    epochs_trained: int
    best_epoch: int
    criterion_order: str


def revisar_datos_faltantes(
    dataframe: pd.DataFrame,
    columnas: list[str],
) -> pd.DataFrame:
    """Return missing-value counts and percentages for the selected columns."""

    columnas_inexistentes = [
        columna for columna in columnas if columna not in dataframe.columns
    ]
    if columnas_inexistentes:
        raise ValueError(
            f"Las siguientes columnas no existen: {columnas_inexistentes}"
        )

    cantidad_faltantes = dataframe[columnas].isna().sum()
    total_filas = max(len(dataframe), 1)
    porcentaje_faltantes = cantidad_faltantes / total_filas * 100

    return pd.DataFrame(
        {
            "cantidad_faltantes": cantidad_faltantes,
            "porcentaje_faltantes": porcentaje_faltantes,
        }
    )


def aplicar_reglas_limpieza(
    df_input: pd.DataFrame,
    variables_categoricas: list[str],
    variables_numericas: list[str],
    valores_faltantes: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Clean categorical and numeric variables for model training."""

    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input debe ser un DataFrame de pandas.")

    columnas_repetidas = set(variables_categoricas) & set(variables_numericas)
    if columnas_repetidas:
        raise ValueError(
            "Las siguientes columnas aparecen como categoricas y numericas: "
            f"{sorted(columnas_repetidas)}"
        )

    df_cleaned = df_input.copy()

    if valores_faltantes is None:
        valores_faltantes = [
            "",
            "na",
            "n/a",
            "null",
            "none",
            "nan",
            "missing",
            "sin dato",
            "<na>",
        ]

    valores_faltantes_normalizados = {
        str(valor).strip().lower() for valor in valores_faltantes
    }

    categoricas_presentes = [
        columna for columna in variables_categoricas if columna in df_cleaned.columns
    ]
    numericas_presentes = [
        columna for columna in variables_numericas if columna in df_cleaned.columns
    ]

    columnas_no_encontradas = [
        columna
        for columna in variables_categoricas + variables_numericas
        if columna not in df_cleaned.columns
    ]
    if columnas_no_encontradas and verbose:
        print("Columnas no encontradas:", columnas_no_encontradas)

    for columna in categoricas_presentes:
        serie_original = df_cleaned[columna]
        serie_texto = serie_original.astype("string").str.strip()

        mascara_textos_faltantes = (
            serie_texto.str.lower().isin(valores_faltantes_normalizados).fillna(False)
        )
        mascara_espacios_vacios = serie_texto.eq("").fillna(False)
        mascara_faltantes = (
            serie_original.isna() | mascara_textos_faltantes | mascara_espacios_vacios
        )

        serie_limpia = serie_texto.mask(mascara_faltantes)
        df_cleaned[columna] = serie_limpia.astype(object).where(serie_limpia.notna(), None)

    for columna in numericas_presentes:
        serie_original = df_cleaned[columna]
        serie_texto = serie_original.astype("string").str.strip()

        mascara_textos_faltantes = (
            serie_texto.str.lower().isin(valores_faltantes_normalizados).fillna(False)
        )
        mascara_espacios_vacios = serie_texto.eq("").fillna(False)
        serie_texto = serie_texto.mask(
            serie_original.isna() | mascara_textos_faltantes | mascara_espacios_vacios
        )

        df_cleaned[columna] = pd.to_numeric(serie_texto, errors="coerce")

    if "Fuel_Type" in df_cleaned.columns:
        fuel_type_normalizado = (
            df_cleaned["Fuel_Type"].astype("string").str.strip().str.lower()
        )
        mascara_electricos = fuel_type_normalizado.eq("electric").fillna(False)
        mascara_hibridos = fuel_type_normalizado.eq("hybrid").fillna(False)
    else:
        mascara_electricos = pd.Series(False, index=df_cleaned.index, dtype=bool)
        mascara_hibridos = pd.Series(False, index=df_cleaned.index, dtype=bool)

    if "Transmission" in df_cleaned.columns:
        cantidad = int(mascara_electricos.sum())
        df_cleaned.loc[mascara_electricos, "Transmission"] = "ST"
        if verbose and cantidad > 0:
            print(f"Transmission='ST' aplicada a {cantidad} vehiculos electricos.")

    if "Engine_Size" in df_cleaned.columns:
        cantidad = int(mascara_electricos.sum())
        df_cleaned.loc[mascara_electricos, "Engine_Size"] = "SM"
        if verbose and cantidad > 0:
            print(f"Engine_Size='SM' aplicada a {cantidad} vehiculos electricos.")

    if "Transmission" in df_cleaned.columns:
        cantidad = int(mascara_hibridos.sum())
        df_cleaned.loc[mascara_hibridos, "Transmission"] = "Automatic"
        if verbose and cantidad > 0:
            print(f"Transmission='Automatic' aplicada a {cantidad} vehiculos hibridos.")

    if "Accident_History" in df_cleaned.columns:
        accident_history = pd.to_numeric(df_cleaned["Accident_History"], errors="coerce")
        mascara_decimales_no_enteros = accident_history.notna() & accident_history.mod(1).ne(0)

        if mascara_decimales_no_enteros.any():
            valores_problematicos = (
                accident_history[mascara_decimales_no_enteros].drop_duplicates().tolist()
            )
            raise ValueError(
                "Accident_History contiene valores decimales no enteros: "
                f"{valores_problematicos}"
            )

        cantidad_faltantes = int(accident_history.isna().sum())
        accident_history_entero = accident_history.astype("Int64")
        df_cleaned["Accident_History"] = (
            accident_history_entero.astype("string").astype(object).where(
                accident_history_entero.notna(),
                None,
            )
        )

        if verbose:
            print(
                f"Accident_History: {cantidad_faltantes} valores faltantes "
                "conservados como nulos y los valores validos convertidos a string."
            )

    for columna in categoricas_presentes:
        serie = df_cleaned[columna]
        df_cleaned[columna] = serie.astype(object).where(pd.notna(serie), None)

    for columna in numericas_presentes:
        df_cleaned[columna] = pd.to_numeric(df_cleaned[columna], errors="coerce")

    return df_cleaned


def resolve_available_columns(
    dataframe: pd.DataFrame,
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return available categorical and numeric columns plus missing columns."""

    available_columns = set(dataframe.columns)
    categorical_available = [
        column for column in categorical_columns if column in available_columns
    ]
    numeric_available = [column for column in numeric_columns if column in available_columns]
    missing_columns = [
        column
        for column in list(categorical_columns) + list(numeric_columns)
        if column not in available_columns
    ]
    return categorical_available, numeric_available, missing_columns


def clean_feature_frame(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """Subset and clean a feature frame using the project rules."""

    selected_columns = categorical_columns + numeric_columns
    if not selected_columns:
        raise ValueError("No hay columnas de entrada disponibles para limpiar.")

    missing_columns = [column for column in selected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    return aplicar_reglas_limpieza(
        dataframe[selected_columns].copy(),
        variables_categoricas=categorical_columns,
        variables_numericas=numeric_columns,
        verbose=verbose,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible with multiple sklearn versions."""

    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> ColumnTransformer:
    """Build the preprocessing pipeline used for training and inference."""

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=None,
                    strategy="constant",
                    fill_value="desconocido",
                ),
            ),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", numeric_pipeline, numeric_columns),
            ("categoricas", categorical_pipeline, categorical_columns),
        ]
    )


def build_regression_model(
    numero_entradas: int,
    neuronas_ocultas: list[int],
    learning_rate: float = 0.001,
) -> Any:
    """Build and compile the Keras regression network."""

    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    model.add(layers.Input(shape=(numero_entradas,)))

    for neuronas in neuronas_ocultas:
        model.add(layers.Dense(neuronas, activation="relu"))

    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def evaluate_regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    """Compute standard regression metrics."""

    y_true_array = np.asarray(list(y_true)).reshape(-1)
    y_pred_array = np.asarray(list(y_pred)).reshape(-1)

    residuals = y_true_array - y_pred_array
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))

    safe_true = np.where(np.abs(y_true_array) < 1e-12, np.nan, y_true_array)
    mape_raw = np.abs((y_true_array - y_pred_array) / safe_true)
    mape = float(np.nanmean(mape_raw) * 100)
    if np.isnan(mape):
        mape = 0.0

    total_sum_squares = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))
    if total_sum_squares == 0:
        r2 = 0.0
    else:
        r2 = float(1.0 - np.sum(residuals**2) / total_sum_squares)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
    }


def split_train_valid_test(
    features: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation and test sets."""

    if test_size <= 0 or validation_size <= 0:
        raise ValueError("test_size y validation_size deben ser mayores que cero.")
    if test_size + validation_size >= 1:
        raise ValueError("La suma de test_size y validation_size debe ser menor que 1.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(features))
    rng.shuffle(indices)

    test_count = int(round(len(features) * test_size))
    valid_count = int(round(len(features) * validation_size))
    test_count = max(test_count, 1)
    valid_count = max(valid_count, 1)

    if test_count + valid_count >= len(features):
        raise ValueError("Las particiones solicitadas dejan sin datos al conjunto de entrenamiento.")

    test_indices = indices[:test_count]
    valid_indices = indices[test_count : test_count + valid_count]
    train_indices = indices[test_count + valid_count :]

    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_valid = features.iloc[valid_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)

    if isinstance(target, pd.DataFrame):
        y_train = target.iloc[train_indices].reset_index(drop=True)
        y_valid = target.iloc[valid_indices].reset_index(drop=True)
        y_test = target.iloc[test_indices].reset_index(drop=True)
    else:
        y_train = target.iloc[train_indices].reset_index(drop=True)
        y_valid = target.iloc[valid_indices].reset_index(drop=True)
        y_test = target.iloc[test_indices].reset_index(drop=True)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def set_global_seed(seed: int) -> None:
    """Set numpy and TensorFlow seeds for reproducibility."""

    import random

    np.random.seed(seed)
    random.seed(seed)

    try:
        from tensorflow import keras

        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def train_single_architecture(
    architecture: list[int],
    crear_modelo: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    epochs: int = 500,
    batch_size: int | None = 32,
    patience: int = 30,
    use_early_stopping: bool = True,
    verbose_training: int = 0,
) -> ArchitectureResult:
    """Train a single architecture and return its metrics."""

    from tensorflow.keras.callbacks import EarlyStopping

    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train y y_train deben tener la misma cantidad de filas.")
    if X_valid.shape[0] != len(y_valid):
        raise ValueError("X_valid y y_valid deben tener la misma cantidad de filas.")
    if not architecture:
        raise ValueError("La arquitectura debe contener al menos una capa oculta.")
    if any(not isinstance(neuronas, int) or neuronas <= 0 for neuronas in architecture):
        raise ValueError(f"Arquitectura invalida: {architecture}")

    numero_entradas = X_train.shape[1]
    batch_size_real = X_train.shape[0] if batch_size is None else min(batch_size, X_train.shape[0])

    modelo = crear_modelo(
        numero_entradas=numero_entradas,
        neuronas_ocultas=architecture,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    history = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size_real,
        callbacks=callbacks,
        verbose=verbose_training,
    )

    pred_train = modelo.predict(X_train, verbose=0).ravel()
    pred_valid = modelo.predict(X_valid, verbose=0).ravel()

    train_metrics = evaluate_regression_metrics(y_train, pred_train)
    valid_metrics = evaluate_regression_metrics(y_valid, pred_valid)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    val_loss_history = history_dict.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss_history) + 1) if val_loss_history else len(history_dict.get("loss", []))

    return ArchitectureResult(
        architecture=tuple(architecture),
        history=history_dict,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        epochs_trained=len(history_dict.get("loss", [])),
        best_epoch=best_epoch,
        model=modelo,
    )


def compare_architectures_regression(
    architectures: list[list[int]],
    crear_modelo: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    epochs: int = 500,
    batch_size: int | None = 32,
    patience: int = 30,
    use_early_stopping: bool = True,
    criterion_order: str = "rmse_valid",
    verbose_training: int = 0,
    show_progress: bool = True,
) -> BestArchitectureResult:
    """Train several architectures and return the best ranked candidate."""

    if not architectures:
        raise ValueError("La lista de arquitecturas no puede estar vacia.")

    if criterion_order not in {
        "mae_train",
        "mse_train",
        "rmse_train",
        "mape_train_pct",
        "r2_train",
        "mae_valid",
        "mse_valid",
        "rmse_valid",
        "mape_valid_pct",
        "r2_valid",
        "val_loss_minima",
        "brecha_mae",
        "brecha_mse",
        "brecha_rmse",
        "brecha_mape_pct",
        "brecha_r2",
    }:
        raise ValueError(f"Criterio de orden invalido: {criterion_order}")

    resultados: list[dict[str, Any]] = []

    for index, architecture in enumerate(architectures, start=1):
        if show_progress:
            print(f"Entrenando arquitectura {index}/{len(architectures)}: {architecture}")

        result = train_single_architecture(
            architecture=architecture,
            crear_modelo=crear_modelo,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            use_early_stopping=use_early_stopping,
            verbose_training=verbose_training,
        )

        brecha_mae = result.valid_metrics["mae"] - result.train_metrics["mae"]
        brecha_mse = result.valid_metrics["mse"] - result.train_metrics["mse"]
        brecha_rmse = result.valid_metrics["rmse"] - result.train_metrics["rmse"]
        brecha_mape_pct = result.valid_metrics["mape_pct"] - result.train_metrics["mape_pct"]
        brecha_r2 = result.train_metrics["r2"] - result.valid_metrics["r2"]
        val_loss_history = result.history.get("val_loss", [])
        val_loss_minima = float(min(val_loss_history)) if val_loss_history else float("nan")

        resultados.append(
            {
                "modelo": index,
                "architecture_tuple": result.architecture,
                "arquitectura": str(list(result.architecture)),
                "neuronas_por_capa": " -> ".join(str(neuronas) for neuronas in result.architecture),
                "capas_ocultas": len(result.architecture),
                "neuronas_totales": sum(result.architecture),
                "epocas_entrenadas": result.epochs_trained,
                "mejor_epoca": result.best_epoch,
                "val_loss_minima": val_loss_minima,
                "mae_train": result.train_metrics["mae"],
                "mse_train": result.train_metrics["mse"],
                "rmse_train": result.train_metrics["rmse"],
                "mape_train_pct": result.train_metrics["mape_pct"],
                "r2_train": result.train_metrics["r2"],
                "mae_valid": result.valid_metrics["mae"],
                "mse_valid": result.valid_metrics["mse"],
                "rmse_valid": result.valid_metrics["rmse"],
                "mape_valid_pct": result.valid_metrics["mape_pct"],
                "r2_valid": result.valid_metrics["r2"],
                "brecha_mae": brecha_mae,
                "brecha_mse": brecha_mse,
                "brecha_rmse": brecha_rmse,
                "brecha_mape_pct": brecha_mape_pct,
                "brecha_r2": brecha_r2,
                "model": result.model,
                "history": result.history,
            }
        )

    tabla_resultados = pd.DataFrame(resultados)

    metricas_mayor_es_mejor = {"r2_train", "r2_valid"}
    orden_ascendente = criterion_order not in metricas_mayor_es_mejor
    tabla_resultados = (
        tabla_resultados.sort_values(by=criterion_order, ascending=orden_ascendente)
        .reset_index(drop=True)
    )
    tabla_resultados.insert(0, "ranking", range(1, len(tabla_resultados) + 1))
    best_row = tabla_resultados.iloc[0]
    return BestArchitectureResult(
        ranking=int(best_row["ranking"]),
        architecture=tuple(best_row["architecture_tuple"]),
        history=best_row["history"],
        model=best_row["model"],
        rmse_valid=float(best_row["rmse_valid"]),
        mae_valid=float(best_row["mae_valid"]),
        r2_valid=float(best_row["r2_valid"]),
        val_loss_minima=float(best_row["val_loss_minima"]),
        epochs_trained=int(best_row["epocas_entrenadas"]),
        best_epoch=int(best_row["mejor_epoca"]),
        criterion_order=criterion_order,
    )


def save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    """Persist a DataFrame as CSV and return the created path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_json(data: Any, output_path: Path) -> Path:
    """Persist JSON-serializable data and return the created path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path


def load_dataset(data_path: Path, sep: str = ",") -> pd.DataFrame:
    """Load the training dataset from disk."""

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {data_path}")
    return pd.read_csv(data_path, sep=sep)


def get_git_commit_sha() -> str | None:
    """Return the current git commit if the repository is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_mlflow_pyfunc_model(
    categorical_columns: list[str],
    numeric_columns: list[str],
    missing_tokens: list[str],
) -> Any:
    """Create an MLflow `pyfunc` wrapper around the trained pricing pipeline.

    The goal of this wrapper is to make inference simple for downstream users:
    they pass a raw `pandas.DataFrame` with the original vehicle columns, and
    MLflow takes care of the rest.

    Internally, the wrapper:
    - validates that the input is a DataFrame;
    - checks that the expected raw columns are present;
    - applies the same cleaning rules used during training;
    - loads the serialized scikit-learn preprocessor;
    - loads the serialized Keras model;
    - returns predictions as a one-column DataFrame.

    Keeping all of that logic in the `pyfunc` layer is important because it
    makes the logged model self-contained and much easier to reuse in notebooks,
    batch jobs, APIs, or the MLflow Model Registry.
    """

    import mlflow.pyfunc

    class AutomobilePricePyFuncModel(mlflow.pyfunc.PythonModel):
        def __init__(
            self,
            categorical_columns: list[str],
            numeric_columns: list[str],
            missing_tokens: list[str],
        ) -> None:
            # The column lists are stored so inference can reproduce the exact
            # same expectations that were used during training.
            self.categorical_columns = categorical_columns
            self.numeric_columns = numeric_columns
            self.missing_tokens = missing_tokens
            self.preprocessor = None
            self.keras_model = None

        def load_context(self, context) -> None:
            # MLflow calls this once when the model is loaded. Here we restore
            # the serialized preprocessing pipeline and the trained neural net
            # from the artifact paths provided by MLflow.
            from tensorflow import keras

            with open(context.artifacts["preprocessor"], "rb") as file:
                self.preprocessor = pickle.load(file)
            self.keras_model = keras.models.load_model(context.artifacts["keras_model"])

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            # MLflow pyfunc receives arbitrary inputs, so we enforce the
            # contract explicitly before trying to transform anything.
            if not isinstance(model_input, pd.DataFrame):
                raise TypeError("El modelo espera un DataFrame de pandas como entrada.")

            expected_columns = self.categorical_columns + self.numeric_columns
            missing_columns = [column for column in expected_columns if column not in model_input.columns]
            if missing_columns:
                raise ValueError(f"Faltan columnas requeridas para inferencia: {missing_columns}")

            # Reutilizamos exactamente la misma limpieza que durante training.
            # Esto evita divergencias entre entrenamiento e inferencia cuando el
            # dataset viene con espacios, nulos representados como texto o tokens
            # ambiguos como "na" o "missing".
            cleaned_input = aplicar_reglas_limpieza(
                model_input[expected_columns].copy(),
                variables_categoricas=self.categorical_columns,
                variables_numericas=self.numeric_columns,
                valores_faltantes=self.missing_tokens,
                verbose=False,
            )

            # El preprocesador transforma columnas crudas en un tensor numérico
            # compatible con el modelo Keras. La prediccion final se devuelve en
            # un DataFrame para mantener una interfaz tabular amigable.
            transformed_input = self.preprocessor.transform(cleaned_input[expected_columns])
            predictions = self.keras_model.predict(transformed_input, verbose=0).ravel()
            return pd.DataFrame({"prediction": predictions})

    return AutomobilePricePyFuncModel(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        missing_tokens=missing_tokens,
    )


def dump_artifacts_to_tempdir(
    preprocessor: Any,
    model: Any,
    temp_dir: Path,
) -> tuple[Path, Path]:
    """Persist the fitted preprocessor and Keras model in a temporary directory.

    MLflow's `pyfunc.log_model` API expects file paths for external artifacts.
    This helper materializes both objects on disk first so they can be attached
    to the logged model package.
    """

    temp_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = temp_dir / "preprocessor.pkl"
    keras_model_path = temp_dir / "keras_model.keras"

    # The preprocessor is serialized with pickle because it is a scikit-learn
    # object. The Keras model is saved in its native `.keras` format so it can
    # be reloaded later without rebuilding the architecture by hand.
    with preprocessor_path.open("wb") as file:
        pickle.dump(preprocessor, file)
    model.save(keras_model_path)
    return preprocessor_path, keras_model_path
