from __future__ import annotations

import pandas as pd

from src.utils import (
    aplicar_reglas_limpieza,
    revisar_datos_faltantes,
    resolve_available_columns,
    split_train_valid_test,
)


def test_revisar_datos_faltantes_reports_counts() -> None:
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

