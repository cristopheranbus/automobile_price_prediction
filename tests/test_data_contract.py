from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DEFAULT_CATEGORICAL_COLUMNS, DEFAULT_NUMERIC_COLUMNS
from src.rnn_price_predict import load_dataset, resolve_available_columns, split_train_valid_test


def test_vehicle_fixture_matches_default_training_contract(vehicle_data) -> None:
    expected_columns = set(DEFAULT_CATEGORICAL_COLUMNS) | set(DEFAULT_NUMERIC_COLUMNS) | {"Selling_Price"}

    assert expected_columns.issubset(vehicle_data.columns)

    categorical, numeric, missing = resolve_available_columns(
        vehicle_data,
        categorical_columns=DEFAULT_CATEGORICAL_COLUMNS,
        numeric_columns=DEFAULT_NUMERIC_COLUMNS,
    )

    assert categorical == list(DEFAULT_CATEGORICAL_COLUMNS)
    assert numeric == list(DEFAULT_NUMERIC_COLUMNS)
    assert missing == []


def test_load_dataset_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("missing.csv"))


def test_split_train_valid_test_rejects_invalid_partitions(vehicle_features, vehicle_target) -> None:
    with pytest.raises(ValueError):
        split_train_valid_test(
            features=vehicle_features,
            target=vehicle_target,
            test_size=0.0,
            validation_size=0.2,
            random_state=42,
        )
