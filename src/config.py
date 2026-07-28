from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "Make",
    "Model",
    "Body_Type",
    "Fuel_Type",
    "Transmission",
    "Engine_Size",
    "Accident_History",
)

DEFAULT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "Year",
    "Mileage",
    "Horsepower",
    "Torque",
    "Owners",
)

DEFAULT_ARCHITECTURES: list[tuple[int, ...]] = [
    (16,),
    (32,),
    (64, 32),
    (128, 64),
    (128, 64, 32),
]


@dataclass(frozen=True)
class DataConfig:
    data_path: Path = Path("data/automobile_dataset.csv")
    target_column: str = "Selling_Price"
    categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS
    numeric_columns: tuple[str, ...] = DEFAULT_NUMERIC_COLUMNS
    missing_tokens: tuple[str, ...] = (
        "",
        "na",
        "n/a",
        "null",
        "none",
        "nan",
        "missing",
        "sin dato",
        "<na>",
    )
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_path"] = str(self.data_path)
        return data


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str = "automobile-price-prediction"
    registered_model_name: str = "automobile-price-predictor"
    tracking_uri: str | None = None
    epochs: int = 300
    batch_size: int | None = None
    patience: int = 30
    learning_rate: float = 0.001
    use_early_stopping: bool = True
    verbose_training: int = 0
    run_name: str | None = None
    architectures: list[tuple[int, ...]] = field(
        default_factory=lambda: list(DEFAULT_ARCHITECTURES)
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    epochs: int
    batch_size: int | None
    patience: int
    learning_rate: float
    use_early_stopping: bool
    verbose_training: int
    architectures: list[tuple[int, ...]]
    criterion_order: str = "rmse_valid"
    run_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_architectures(raw_architectures: Any) -> list[tuple[int, ...]]:
    if raw_architectures is None:
        return list(DEFAULT_ARCHITECTURES)

    normalized: list[tuple[int, ...]] = []
    for architecture in raw_architectures:
        if isinstance(architecture, (list, tuple)):
            normalized.append(tuple(int(value) for value in architecture))
        else:
            normalized.append((int(architecture),))
    return normalized


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""

    if not config_path.exists():
        return {}

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency issue
        raise ModuleNotFoundError(
            "PyYAML no esta instalado. Ejecuta `pip install -r requirements.txt`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("El archivo YAML de configuracion debe contener un mapping raiz.")

    return loaded


def build_data_config(config_data: dict[str, Any] | None = None) -> DataConfig:
    """Build a DataConfig from a YAML-derived dictionary."""

    config_data = config_data or {}
    return DataConfig(
        data_path=Path(config_data.get("data_path", DataConfig().data_path)),
        target_column=config_data.get("target_column", DataConfig().target_column),
        categorical_columns=tuple(
            config_data.get("categorical_columns", DataConfig().categorical_columns)
        ),
        numeric_columns=tuple(
            config_data.get("numeric_columns", DataConfig().numeric_columns)
        ),
        missing_tokens=tuple(
            config_data.get("missing_tokens", DataConfig().missing_tokens)
        ),
        test_size=float(config_data.get("test_size", DataConfig().test_size)),
        validation_size=float(
            config_data.get("validation_size", DataConfig().validation_size)
        ),
        random_state=int(config_data.get("random_state", DataConfig().random_state)),
    )


def build_training_config(config_data: dict[str, Any] | None = None) -> TrainingConfig:
    """Build a TrainingConfig from a YAML-derived dictionary."""

    config_data = config_data or {}
    return TrainingConfig(
        experiment_name=config_data.get(
            "experiment_name", TrainingConfig().experiment_name
        ),
        registered_model_name=config_data.get(
            "registered_model_name", TrainingConfig().registered_model_name
        ),
        tracking_uri=config_data.get("tracking_uri", TrainingConfig().tracking_uri),
        epochs=int(config_data.get("epochs", TrainingConfig().epochs)),
        batch_size=(
            None
            if config_data.get("batch_size", TrainingConfig().batch_size) is None
            else int(config_data.get("batch_size", TrainingConfig().batch_size))
        ),
        patience=int(config_data.get("patience", TrainingConfig().patience)),
        learning_rate=float(
            config_data.get("learning_rate", TrainingConfig().learning_rate)
        ),
        use_early_stopping=bool(
            config_data.get(
                "use_early_stopping", TrainingConfig().use_early_stopping
            )
        ),
        verbose_training=int(
            config_data.get("verbose_training", TrainingConfig().verbose_training)
        ),
        run_name=config_data.get("run_name", TrainingConfig().run_name),
        architectures=_normalize_architectures(
            config_data.get("architectures", DEFAULT_ARCHITECTURES)
        ),
    )


def build_experiment_configs(
    config_data: dict[str, Any] | None = None,
    base_training: TrainingConfig | None = None,
) -> list[ExperimentConfig]:
    """Build one or more experiment configs from YAML data."""

    config_data = config_data or {}
    base_training = base_training or TrainingConfig()
    raw_experiments = config_data.get("experiments")

    if raw_experiments is None:
        raw_experiments = [
            {
                "name": config_data.get("run_name") or "default_experiment",
                "epochs": config_data.get("epochs", base_training.epochs),
                "batch_size": config_data.get("batch_size", base_training.batch_size),
                "patience": config_data.get("patience", base_training.patience),
                "learning_rate": config_data.get(
                    "learning_rate", base_training.learning_rate
                ),
                "use_early_stopping": config_data.get(
                    "use_early_stopping", base_training.use_early_stopping
                ),
                "verbose_training": config_data.get(
                    "verbose_training", base_training.verbose_training
                ),
                "architectures": config_data.get(
                    "architectures", base_training.architectures
                ),
                "criterion_order": config_data.get("criterion_order", "rmse_valid"),
                "run_name": config_data.get("run_name", None),
            }
        ]

    if isinstance(raw_experiments, dict):
        raw_experiments = [raw_experiments]

    experiments: list[ExperimentConfig] = []
    for index, experiment_data in enumerate(raw_experiments, start=1):
        if not isinstance(experiment_data, dict):
            raise TypeError(
                "Cada experimento debe ser un mapping YAML con sus parametros."
            )

        architectures = _normalize_architectures(
            experiment_data.get("architectures", base_training.architectures)
        )

        experiment_name = experiment_data.get(
            "name",
            experiment_data.get(
                "run_name",
                f"experiment_{index}",
            ),
        )

        experiments.append(
            ExperimentConfig(
                name=str(experiment_name),
                epochs=int(experiment_data.get("epochs", base_training.epochs)),
                batch_size=(
                    None
                    if experiment_data.get("batch_size", base_training.batch_size) is None
                    else int(experiment_data.get("batch_size", base_training.batch_size))
                ),
                patience=int(experiment_data.get("patience", base_training.patience)),
                learning_rate=float(
                    experiment_data.get("learning_rate", base_training.learning_rate)
                ),
                use_early_stopping=bool(
                    experiment_data.get(
                        "use_early_stopping",
                        base_training.use_early_stopping,
                    )
                ),
                verbose_training=int(
                    experiment_data.get(
                        "verbose_training", base_training.verbose_training
                    )
                ),
                architectures=architectures,
                criterion_order=str(
                    experiment_data.get("criterion_order", "rmse_valid")
                ),
                run_name=experiment_data.get("run_name", None),
            )
        )

    return experiments
