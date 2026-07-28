from __future__ import annotations

from pathlib import Path

from src.config import (
    build_data_config,
    build_experiment_configs,
    build_training_config,
)


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
            "architectures": [[16], [32, 16], 64],
        }
    )

    assert config.experiment_name == "demo"
    assert config.epochs == 50
    assert config.use_early_stopping is False
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
