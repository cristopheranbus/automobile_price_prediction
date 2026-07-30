from __future__ import annotations

import pandas as pd
import pytest

from src.config import DEFAULT_CATEGORICAL_COLUMNS, DEFAULT_NUMERIC_COLUMNS


@pytest.fixture
def vehicle_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Make": ["A", "B", "A", "B", "A", "B"],
            "Model": ["M1", "M2", "M1", "M2", "M1", "M2"],
            "Body_Type": ["Sedan", "SUV", "Sedan", "SUV", "Sedan", "SUV"],
            "Fuel_Type": ["Gas", "Gas", "Hybrid", "Gas", "Gas", "Hybrid"],
            "Transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual", "Automatic"],
            "Engine_Size": ["1.8", "2.0", "1.8", "2.0", "1.8", "2.0"],
            "Accident_History": [0, 1, 0, 1, 0, 1],
            "Year": [2020, 2021, 2020, 2021, 2020, 2021],
            "Mileage": [10000, 20000, 12000, 22000, 13000, 21000],
            "Horsepower": [120, 130, 125, 135, 128, 138],
            "Torque": [200, 210, 205, 215, 208, 218],
            "Owners": [1, 2, 1, 2, 1, 2],
            "Selling_Price": [100, 110, 105, 115, 108, 118],
        }
    )


@pytest.fixture
def vehicle_features(vehicle_data: pd.DataFrame) -> pd.DataFrame:
    return vehicle_data.drop(columns=["Selling_Price"]).copy()


@pytest.fixture
def vehicle_target(vehicle_data: pd.DataFrame) -> pd.Series:
    return vehicle_data["Selling_Price"].copy()


@pytest.fixture
def architecture_suite() -> list[list[int]]:
    return [[16], [32, 16]]


@pytest.fixture
def default_feature_columns() -> tuple[list[str], list[str]]:
    return list(DEFAULT_CATEGORICAL_COLUMNS), list(DEFAULT_NUMERIC_COLUMNS)
