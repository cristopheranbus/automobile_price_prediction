from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rnn_price_predict import build_mlflow_pyfunc_model


class DummyPreprocessor:
    def __init__(self) -> None:
        self.last_input = None

    def transform(self, dataframe):
        self.last_input = dataframe.copy()
        return np.ones((len(dataframe), 3), dtype=float)


class DummyKerasModel:
    def __init__(self) -> None:
        self.last_input = None

    def predict(self, X, verbose: int = 0):  # noqa: N803
        self.last_input = X
        values = np.arange(len(X), dtype=float).reshape(-1, 1)
        return values


def test_pyfunc_predict_returns_dataframe(vehicle_features, default_feature_columns) -> None:
    categorical_columns, numeric_columns = default_feature_columns
    model = build_mlflow_pyfunc_model(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        missing_tokens=["", "na", "none"],
    )
    model.preprocessor = DummyPreprocessor()
    model.keras_model = DummyKerasModel()

    result = model.predict(None, vehicle_features.head(2))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["prediction"]
    assert len(result) == 2
    assert result.iloc[0, 0] == pytest.approx(0.0)
    assert result.iloc[1, 0] == pytest.approx(1.0)


def test_pyfunc_predict_rejects_missing_columns(vehicle_features, default_feature_columns) -> None:
    categorical_columns, numeric_columns = default_feature_columns
    model = build_mlflow_pyfunc_model(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        missing_tokens=["", "na", "none"],
    )
    model.preprocessor = DummyPreprocessor()
    model.keras_model = DummyKerasModel()

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        model.predict(None, vehicle_features.drop(columns=["Mileage"]).head(2))


def test_pyfunc_predict_rejects_non_dataframe_input(default_feature_columns) -> None:
    categorical_columns, numeric_columns = default_feature_columns
    model = build_mlflow_pyfunc_model(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        missing_tokens=["", "na", "none"],
    )
    model.preprocessor = DummyPreprocessor()
    model.keras_model = DummyKerasModel()

    with pytest.raises(TypeError, match="DataFrame de pandas"):
        model.predict(None, {"not": "a dataframe"})
