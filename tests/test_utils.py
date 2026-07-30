from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rnn_price_predict import (
    ArchitectureResult,
    aplicar_reglas_limpieza,
    build_preprocessor,
    compare_architectures_regression,
    evaluate_regression_metrics,
    resolve_available_columns,
    split_train_valid_test,
)


def test_revisar_datos_faltantes_reports_counts() -> None:
    from src.rnn_price_predict import revisar_datos_faltantes

    df = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": ["x", "y", None],
        }
    )

    result = revisar_datos_faltantes(df, ["a", "b"])

    assert result.loc["a", "cantidad_faltantes"] == 1
    assert result.loc["b", "cantidad_faltantes"] == 1


def test_aplicar_reglas_limpieza_applies_vehicle_rules() -> None:
    df = pd.DataFrame(
        {
            "Fuel_Type": ["Electric", "Hybrid", "Gas"],
            "Transmission": ["Manual", "Manual", "Manual"],
            "Engine_Size": ["Large", "Large", "Large"],
            "Accident_History": ["1", "2", None],
            "Year": ["2020", "2021", "2022"],
        }
    )

    cleaned = aplicar_reglas_limpieza(
        df,
        variables_categoricas=["Fuel_Type", "Transmission", "Engine_Size", "Accident_History"],
        variables_numericas=["Year"],
    )

    assert cleaned.loc[0, "Transmission"] == "ST"
    assert cleaned.loc[0, "Engine_Size"] == "SM"
    assert cleaned.loc[1, "Transmission"] == "Automatic"
    assert cleaned["Year"].dtype.kind in {"i", "f"}


def test_resolve_available_columns_filters_missing_columns() -> None:
    df = pd.DataFrame({"Make": ["A"], "Year": [2020]})

    categorical, numeric, missing = resolve_available_columns(
        df,
        categorical_columns=["Make", "Fuel_Type"],
        numeric_columns=["Year", "Mileage"],
    )

    assert categorical == ["Make"]
    assert numeric == ["Year"]
    assert set(missing) == {"Fuel_Type", "Mileage"}


def test_split_train_valid_test_uses_all_rows() -> None:
    features = pd.DataFrame({"x": range(20)})
    target = pd.Series(range(20))

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_train_valid_test(
        features=features,
        target=target,
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
    )

    assert len(x_train) + len(x_valid) + len(x_test) == len(features)
    assert len(y_train) + len(y_valid) + len(y_test) == len(target)


def test_evaluate_regression_metrics_returns_expected_values() -> None:
    metrics = evaluate_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])

    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["mse"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["mape_pct"] == pytest.approx((0.0 + 0.0 + 33.3333333333) / 3.0, rel=1e-3)
    assert metrics["r2"] == pytest.approx(0.5)


def test_build_preprocessor_handles_unknown_categories(vehicle_features, default_feature_columns) -> None:
    categorical_columns, numeric_columns = default_feature_columns
    preprocessor = build_preprocessor(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    cleaned = aplicar_reglas_limpieza(
        vehicle_features,
        variables_categoricas=categorical_columns,
        variables_numericas=numeric_columns,
    )
    preprocessor.fit(cleaned)

    unseen = cleaned.head(2).copy()
    unseen.loc[0, "Make"] = "Desconocida"
    transformed = preprocessor.transform(unseen)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(numeric_columns)


@pytest.mark.parametrize(
    ("criterion_order", "expected_architecture"),
    [
        ("rmse_valid", (16,)),
        ("r2_valid", (32, 16)),
    ],
)
def test_compare_architectures_respects_criterion_order(
    monkeypatch,
    vehicle_features,
    vehicle_target,
    architecture_suite,
    criterion_order,
    expected_architecture,
) -> None:
    class DummyModel:
        def __init__(self, label: str) -> None:
            self.label = label

        def predict(self, X, verbose: int = 0):  # noqa: N803
            return np.full((len(X), 1), 0.0, dtype=float)

    def fake_train_single_architecture(
        architecture,
        crear_modelo,
        X_train,
        y_train,
        X_valid,
        y_valid,
        **kwargs,
    ):
        if tuple(architecture) == (16,):
            return ArchitectureResult(
                architecture=(16,),
                history={"loss": [2.0], "val_loss": [1.2]},
                train_metrics={"mae": 4.0, "mse": 16.0, "rmse": 4.0, "mape_pct": 10.0, "r2": 0.2},
                valid_metrics={"mae": 3.0, "mse": 9.0, "rmse": 1.0, "mape_pct": 9.0, "r2": 0.4},
                epochs_trained=1,
                best_epoch=1,
                model=DummyModel("small"),
            )

        return ArchitectureResult(
            architecture=(32, 16),
            history={"loss": [1.5], "val_loss": [0.8]},
            train_metrics={"mae": 2.0, "mse": 4.0, "rmse": 2.0, "mape_pct": 6.0, "r2": 0.5},
            valid_metrics={"mae": 1.0, "mse": 1.0, "rmse": 2.0, "mape_pct": 5.0, "r2": 0.9},
            epochs_trained=1,
            best_epoch=1,
            model=DummyModel("large"),
        )

    monkeypatch.setattr(
        "src.rnn_price_predict.train_single_architecture",
        fake_train_single_architecture,
    )

    result = compare_architectures_regression(
        architectures=architecture_suite,
        crear_modelo=lambda numero_entradas, neuronas_ocultas: DummyModel("factory"),
        X_train=vehicle_features.iloc[:3],
        y_train=vehicle_target.iloc[:3],
        X_valid=vehicle_features.iloc[3:5],
        y_valid=vehicle_target.iloc[3:5],
        criterion_order=criterion_order,
        show_progress=False,
    )

    assert result.ranking == 1
    assert result.architecture == expected_architecture
    assert result.criterion_order == criterion_order
    assert result.model.label in {"small", "large"}
    assert result.history["val_loss"]


def test_compare_architectures_rejects_invalid_criterion_order(vehicle_features, vehicle_target) -> None:
    with pytest.raises(ValueError, match="Criterio de orden invalido"):
        compare_architectures_regression(
            architectures=[[16]],
            crear_modelo=lambda numero_entradas, neuronas_ocultas: object(),
            X_train=vehicle_features.iloc[:3],
            y_train=vehicle_target.iloc[:3],
            X_valid=vehicle_features.iloc[3:5],
            y_valid=vehicle_target.iloc[3:5],
            criterion_order="not_a_metric",
            show_progress=False,
        )
