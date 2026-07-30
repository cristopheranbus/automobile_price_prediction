from __future__ import annotations

from pathlib import Path

from src.config import (
    build_data_config,
    build_experiment_configs,
    build_training_config,
)


def test_build_configs_from_empty_mapping_uses_defaults() -> None:
    data_config = build_data_config({})
    training_config = build_training_config({})

    assert data_config.data_path == Path("data/automobile_dataset.csv")
    assert data_config.target_column == "Selling_Price"
    assert training_config.experiment_name == "automobile-price-prediction"
    assert training_config.registered_model_name == "automobile-price-predictor"
    assert training_config.tracking_dir == Path("mlruns")
    assert training_config.model_artifact_path == "model"
    assert training_config.config_artifact_path == "config/config.json"
    assert training_config.history_artifact_path == "reports/final_training_history.json"
    assert training_config.summary_artifact_path == "reports/experiment_summary.json"
    assert training_config.reports_artifact_dir == "reports"
    assert training_config.architectures


def test_build_data_config_from_mapping() -> None:
    config = build_data_config(
        {
            "data_path": "data/sample.csv",
            "target_column": "Price",
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 7,
        }
    )

    assert config.data_path == Path("data/sample.csv")
    assert config.target_column == "Price"
    assert config.test_size == 0.2
    assert config.validation_size == 0.1
    assert config.random_state == 7


def test_build_training_config_normalizes_architectures() -> None:
    config = build_training_config(
        {
            "experiment_name": "demo",
            "epochs": 50,
            "use_early_stopping": False,
            "tracking_dir": "mlruns",
            "model_artifact_path": "artifacts/model",
            "config_artifact_path": "artifacts/config.json",
            "history_artifact_path": "artifacts/history.json",
            "summary_artifact_path": "artifacts/summary.json",
            "reports_artifact_dir": "artifacts/reports",
            "architectures": [[16], [32, 16], 64],
        }
    )

    assert config.experiment_name == "demo"
    assert config.epochs == 50
    assert config.use_early_stopping is False
    assert config.tracking_dir == Path("mlruns")
    assert config.model_artifact_path == "artifacts/model"
    assert config.config_artifact_path == "artifacts/config.json"
    assert config.history_artifact_path == "artifacts/history.json"
    assert config.summary_artifact_path == "artifacts/summary.json"
    assert config.reports_artifact_dir == "artifacts/reports"
    assert config.architectures == [(16,), (32, 16), (64,)]


def test_build_experiment_configs_from_suite() -> None:
    base_training = build_training_config(
        {
            "experiment_name": "suite",
            "epochs": 100,
            "learning_rate": 0.01,
            "architectures": [[8], [16, 8]],
        }
    )

    experiments = build_experiment_configs(
        {
            "experiments": [
                {
                    "name": "small",
                    "epochs": 50,
                    "architectures": [[4], [8]],
                },
                {
                    "name": "default-inherited",
                },
            ]
        },
        base_training=base_training,
    )

    assert len(experiments) == 2
    assert experiments[0].name == "small"
    assert experiments[0].epochs == 50
    assert experiments[0].architectures == [(4,), (8,)]
    assert experiments[1].name == "default-inherited"
    assert experiments[1].epochs == 100
    assert experiments[1].learning_rate == 0.01
    assert experiments[1].architectures == [(8,), (16, 8)]
