from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    build_data_config,
    build_experiment_configs,
    build_training_config,
    load_yaml_config,
)
from src.utils import (
    build_mlflow_pyfunc_model,
    build_preprocessor,
    build_regression_model,
    compare_architectures_regression,
    clean_feature_frame,
    dump_artifacts_to_tempdir,
    evaluate_regression_metrics,
    get_git_commit_sha,
    load_dataset,
    resolve_available_columns,
    save_dataframe,
    set_global_seed,
    split_train_valid_test,
)


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train.yaml"),
        help="YAML configuration file.",
    )
    bootstrap_args, remaining_args = bootstrap_parser.parse_known_args()

    yaml_config = load_yaml_config(bootstrap_args.config)
    data_defaults = build_data_config(yaml_config.get("data", {}))
    training_defaults = build_training_config(yaml_config.get("training", {}))

    parser = argparse.ArgumentParser(
        parents=[bootstrap_parser],
        description="Train the automobile price prediction model with MLflow tracking."
    )
    parser.add_argument("--data-path", type=Path, default=data_defaults.data_path)
    parser.add_argument("--tracking-uri", type=str, default=training_defaults.tracking_uri)
    parser.add_argument("--experiment-name", type=str, default=training_defaults.experiment_name)
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=training_defaults.registered_model_name,
    )
    parser.add_argument("--epochs", type=int, default=training_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument("--patience", type=int, default=training_defaults.patience)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=training_defaults.learning_rate,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=data_defaults.test_size,
        help="Fraction of the data reserved for the final test set.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=data_defaults.validation_size,
        help="Fraction of the data reserved for validation.",
    )
    parser.add_argument("--random-state", type=int, default=data_defaults.random_state)
    parser.add_argument(
        "--run-name",
        type=str,
        default=training_defaults.run_name,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="Do not register the final model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--verbose-training",
        type=int,
        default=training_defaults.verbose_training,
    )
    parser.add_argument(
        "--use-early-stopping",
        dest="use_early_stopping",
        action="store_true",
        default=training_defaults.use_early_stopping,
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="use_early_stopping",
        action="store_false",
    )

    parser.set_defaults(config=bootstrap_args.config)
    args = parser.parse_args(remaining_args)
    args.config = bootstrap_args.config
    return args


def create_model_factory(learning_rate: float) -> Any:
    def crear_modelo(numero_entradas: int, neuronas_ocultas: list[int]) -> Any:
        return build_regression_model(
            numero_entradas=numero_entradas,
            neuronas_ocultas=neuronas_ocultas,
            learning_rate=learning_rate,
        )

    return crear_modelo


def _parse_architecture_text(raw_architecture: str) -> list[int]:
    cleaned = raw_architecture.strip().strip("[]")
    if not cleaned:
        return []
    return [int(item.strip()) for item in cleaned.split(",") if item.strip()]


def run_training_pipeline(args: argparse.Namespace) -> None:
    import mlflow
    import mlflow.pyfunc
    from mlflow.models.signature import infer_signature

    yaml_config = load_yaml_config(args.config)
    data_section = yaml_config.get("data", {})
    training_section = yaml_config.get("training", {})

    data_config = build_data_config(
        {
            **data_section,
            "data_path": args.data_path,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "random_state": args.random_state,
        }
    )
    training_config = build_training_config(
        {
            **training_section,
            "experiment_name": args.experiment_name,
            "registered_model_name": args.registered_model_name,
            "tracking_uri": args.tracking_uri,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "verbose_training": args.verbose_training,
            "run_name": args.run_name,
            "use_early_stopping": args.use_early_stopping,
        }
    )
    experiment_configs = build_experiment_configs(training_section, training_config)

    set_global_seed(data_config.random_state)

    if training_config.tracking_uri:
        mlflow.set_tracking_uri(training_config.tracking_uri)
    mlflow.set_experiment(training_config.experiment_name)

    raw_df = load_dataset(data_config.data_path)
    categorical_columns, numeric_columns, missing_columns = resolve_available_columns(
        raw_df,
        data_config.categorical_columns,
        data_config.numeric_columns,
    )

    if not categorical_columns and not numeric_columns:
        raise ValueError("No se encontraron columnas de entrada compatibles en el dataset.")

    if data_config.target_column not in raw_df.columns:
        raise ValueError(f"No existe la columna objetivo '{data_config.target_column}'.")

    feature_columns = categorical_columns + numeric_columns
    raw_features = raw_df[feature_columns].copy()
    raw_target = raw_df[data_config.target_column].copy()

    x_train_raw, x_valid_raw, x_test_raw, y_train, y_valid, y_test = split_train_valid_test(
        features=raw_features,
        target=raw_target,
        test_size=data_config.test_size,
        validation_size=data_config.validation_size,
        random_state=data_config.random_state,
    )

    x_train_clean = clean_feature_frame(
        x_train_raw,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        verbose=False,
    )
    x_valid_clean = clean_feature_frame(
        x_valid_raw,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        verbose=False,
    )
    x_test_clean = clean_feature_frame(
        x_test_raw,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        verbose=False,
    )

    preprocessor = build_preprocessor(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    x_train_processed = preprocessor.fit_transform(x_train_clean)
    x_valid_processed = preprocessor.transform(x_valid_clean)

    with mlflow.start_run(run_name=training_config.run_name):
        git_commit = get_git_commit_sha()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("config_path", str(args.config))

        mlflow.set_tag("data_path", str(data_config.data_path))
        mlflow.set_tag("data_rows", int(len(raw_df)))
        mlflow.set_tag("feature_columns", ",".join(feature_columns))
        mlflow.log_params(
            {
                "target_column": data_config.target_column,
                "random_state": data_config.random_state,
                "test_size": data_config.test_size,
                "validation_size": data_config.validation_size,
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size or "full_batch",
                "patience": training_config.patience,
                "learning_rate": training_config.learning_rate,
                "use_early_stopping": training_config.use_early_stopping,
            }
        )

        mlflow.log_dict(
            {
                "categorical_columns": categorical_columns,
                "numeric_columns": numeric_columns,
                "missing_columns": missing_columns,
                "data_config": data_config.as_dict(),
                "training_config": training_config.as_dict(),
            },
            "config/config.json",
        )

        baseline_value = float(np.median(y_train))
        baseline_predictions = np.full(shape=len(y_test), fill_value=baseline_value)
        baseline_metrics = evaluate_regression_metrics(y_test, baseline_predictions)
        mlflow.log_metrics({f"baseline_{key}": value for key, value in baseline_metrics.items()})

        experiment_results: list[dict[str, Any]] = []

        for experiment_config in experiment_configs:
            comparison_factory = create_model_factory(experiment_config.learning_rate)

            with mlflow.start_run(run_name=experiment_config.run_name or experiment_config.name, nested=True):
                mlflow.set_tag("experiment_name", experiment_config.name)
                mlflow.log_params(
                    {
                        "epochs": experiment_config.epochs,
                        "batch_size": experiment_config.batch_size or "full_batch",
                        "patience": experiment_config.patience,
                        "learning_rate": experiment_config.learning_rate,
                        "use_early_stopping": experiment_config.use_early_stopping,
                        "verbose_training": experiment_config.verbose_training,
                        "criterion_order": experiment_config.criterion_order,
                        "architectures": str(experiment_config.architectures),
                    }
                )

                comparison_table = compare_architectures_regression(
                    architectures=experiment_config.architectures,
                    crear_modelo=comparison_factory,
                    X_train=x_train_processed,
                    y_train=y_train,
                    X_valid=x_valid_processed,
                    y_valid=y_valid,
                    epochs=experiment_config.epochs,
                    batch_size=experiment_config.batch_size,
                    patience=experiment_config.patience,
                    use_early_stopping=experiment_config.use_early_stopping,
                    criterion_order=experiment_config.criterion_order,
                    verbose_training=experiment_config.verbose_training,
                    show_progress=True,
                )

                comparison_artifact = save_dataframe(
                    comparison_table.drop(columns=["model", "history"]),
                    Path(tempfile.gettempdir())
                    / f"automobile_model_comparison_{mlflow.active_run().info.run_id}.csv",
                )
                mlflow.log_artifact(str(comparison_artifact), artifact_path="reports")

                best_row = comparison_table.iloc[0]
                best_architecture = _parse_architecture_text(best_row["arquitectura"])
                mlflow.log_param("best_architecture", str(best_architecture))
                mlflow.log_metric("best_valid_rmse", float(best_row["rmse_valid"]))
                mlflow.log_metric("best_valid_mae", float(best_row["mae_valid"]))
                mlflow.log_metric("best_valid_r2", float(best_row["r2_valid"]))

                final_features = pd.concat([x_train_raw, x_valid_raw], axis=0)
                final_target = pd.concat([y_train, y_valid], axis=0)
                final_features_clean = clean_feature_frame(
                    final_features,
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                    verbose=False,
                )
                final_preprocessor = build_preprocessor(
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                )
                final_preprocessor.fit(final_features_clean)
                final_features_processed = final_preprocessor.transform(final_features_clean)

                final_model = build_regression_model(
                    numero_entradas=final_features_processed.shape[1],
                    neuronas_ocultas=best_architecture,
                    learning_rate=experiment_config.learning_rate,
                )

                history = final_model.fit(
                    final_features_processed,
                    final_target,
                    epochs=experiment_config.epochs,
                    batch_size=(
                        final_features_processed.shape[0]
                        if experiment_config.batch_size is None
                        else min(
                            experiment_config.batch_size,
                            final_features_processed.shape[0],
                        )
                    ),
                    verbose=experiment_config.verbose_training,
                )

                x_test_final_processed = final_preprocessor.transform(x_test_clean)
                test_predictions = final_model.predict(
                    x_test_final_processed,
                    verbose=0,
                ).ravel()
                test_metrics = evaluate_regression_metrics(y_test, test_predictions)
                mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

                history_dict = {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                }
                mlflow.log_dict(history_dict, "reports/final_training_history.json")

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    preprocessor_path, keras_model_path = dump_artifacts_to_tempdir(
                        preprocessor=final_preprocessor,
                        model=final_model,
                        temp_dir=temp_dir,
                    )
                    input_example = x_test_raw.head(5)
                    signature = infer_signature(
                        input_example,
                        pd.DataFrame(
                            {
                                "prediction": final_model.predict(
                                    x_test_final_processed[:5],
                                    verbose=0,
                                ).ravel()
                            }
                        ),
                    )
                    pyfunc_model = build_mlflow_pyfunc_model(
                        categorical_columns=categorical_columns,
                        numeric_columns=numeric_columns,
                        missing_tokens=list(data_config.missing_tokens),
                    )

                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=pyfunc_model,
                        artifacts={
                            "preprocessor": str(preprocessor_path),
                            "keras_model": str(keras_model_path),
                        },
                        input_example=input_example,
                        signature=signature,
                    )

                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                    mlflow.log_param("model_uri", model_uri)

                experiment_results.append(
                    {
                        "experiment_name": experiment_config.name,
                        "run_id": mlflow.active_run().info.run_id,
                        "model_uri": model_uri,
                        "best_architecture": str(best_architecture),
                        "best_valid_rmse": float(best_row["rmse_valid"]),
                        "best_valid_mae": float(best_row["mae_valid"]),
                        "best_valid_r2": float(best_row["r2_valid"]),
                        "test_rmse": float(test_metrics["rmse"]),
                        "test_mae": float(test_metrics["mae"]),
                        "test_r2": float(test_metrics["r2"]),
                        "criterion_order": experiment_config.criterion_order,
                    }
                )

                mlflow.set_tag("status", "completed")

        experiment_results_df = pd.DataFrame(experiment_results)
        summary_artifact = save_dataframe(
            experiment_results_df,
            Path(tempfile.gettempdir())
            / f"automobile_experiment_summary_{mlflow.active_run().info.run_id}.csv",
        )
        mlflow.log_artifact(str(summary_artifact), artifact_path="reports")
        mlflow.log_dict(
            experiment_results_df.to_dict(orient="records"),
            "reports/experiment_summary.json",
        )

        if experiment_results_df.empty:
            raise RuntimeError("No se pudo completar ningun experimento.")

        winner_row = experiment_results_df.sort_values(
            by="best_valid_rmse",
            ascending=True,
        ).iloc[0]

        mlflow.log_param("winning_experiment", winner_row["experiment_name"])
        mlflow.log_param("winning_model_uri", winner_row["model_uri"])
        mlflow.log_metric("winning_valid_rmse", float(winner_row["best_valid_rmse"]))
        mlflow.log_metric("winning_test_rmse", float(winner_row["test_rmse"]))

        if not args.no_registration and training_config.registered_model_name:
            try:
                mlflow.register_model(
                    str(winner_row["model_uri"]),
                    training_config.registered_model_name,
                )
                mlflow.set_tag("registry_status", "registered")
            except Exception as exc:
                mlflow.set_tag("registry_status", f"registration_failed: {exc}")

        mlflow.set_tag("status", "completed")


def main() -> None:
    args = parse_args()
    run_training_pipeline(args)


if __name__ == "__main__":
    main()
