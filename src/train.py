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
from src.rnn_price_predict import (
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
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=training_defaults.tracking_dir,
        help="Local folder used as MLflow file-based tracking backend when no URI is provided.",
    )
    parser.add_argument(
        "--model-artifact-path",
        type=str,
        default=training_defaults.model_artifact_path,
        help="Artifact path inside each MLflow run where the final model will be stored.",
    )
    parser.add_argument(
        "--config-artifact-path",
        type=str,
        default=training_defaults.config_artifact_path,
        help="Artifact path used to store the resolved configuration snapshot.",
    )
    parser.add_argument(
        "--history-artifact-path",
        type=str,
        default=training_defaults.history_artifact_path,
        help="Artifact path used to store the selected model training history.",
    )
    parser.add_argument(
        "--summary-artifact-path",
        type=str,
        default=training_defaults.summary_artifact_path,
        help="Artifact path used to store the experiment summary JSON.",
    )
    parser.add_argument(
        "--reports-artifact-dir",
        type=str,
        default=training_defaults.reports_artifact_dir,
        help="Artifact directory used for CSV reports logged to MLflow.",
    )
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


def resolve_mlflow_tracking_uri(
    tracking_uri: str | None,
    tracking_dir: Path | None,
) -> str | None:
    """Resolve the MLflow tracking backend from an explicit URI or local folder.

    Precedence:
    1. `tracking_uri` wins, because it may point to a remote MLflow server or
       another custom backend.
    2. If no URI is provided, `tracking_dir` is converted into a `file:///...`
       URI so MLflow writes runs to that local folder.
    3. If neither is present, MLflow keeps its default local behavior.
    """

    if tracking_uri:
        return tracking_uri

    if tracking_dir is None:
        return None

    resolved_dir = tracking_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir.as_uri()


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
            "tracking_dir": args.tracking_dir,
            "model_artifact_path": args.model_artifact_path,
            "config_artifact_path": args.config_artifact_path,
            "history_artifact_path": args.history_artifact_path,
            "summary_artifact_path": args.summary_artifact_path,
            "reports_artifact_dir": args.reports_artifact_dir,
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

    effective_tracking_uri = resolve_mlflow_tracking_uri(
        training_config.tracking_uri,
        training_config.tracking_dir,
    )

    # La configuracion de MLflow se aplica una sola vez al inicio del pipeline.
    # Si el usuario indico un tracking URI, los runs se enviaran a ese servidor.
    # Si no lo hizo pero definio un tracking_dir, se usa un backend local basado
    # en archivos dentro de esa carpeta. Si no existe ninguno, MLflow mantiene
    # su comportamiento local por defecto.
    if effective_tracking_uri:
        mlflow.set_tracking_uri(effective_tracking_uri)

    # El experimento agrupa todas las corridas bajo un mismo nombre logico.
    # Esto permite comparar ejecuciones historicas, filtrar por proyecto y
    # navegar facilmente entre runs en la interfaz de MLflow.
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

    # Este es el run principal. Representa la ejecucion completa del pipeline:
    # carga de datos, split, comparacion de arquitecturas, evaluacion final,
    # logging de artefactos y, opcionalmente, registro en el Model Registry.
    with mlflow.start_run(run_name=training_config.run_name):
        mlflow.set_tag("mlflow_tracking_uri", effective_tracking_uri or "default-local")
        mlflow.set_tag("mlflow_tracking_dir", str(training_config.tracking_dir))
        mlflow.set_tag("mlflow_model_artifact_path", training_config.model_artifact_path)
        mlflow.set_tag("mlflow_config_artifact_path", training_config.config_artifact_path)
        mlflow.set_tag("mlflow_history_artifact_path", training_config.history_artifact_path)
        mlflow.set_tag("mlflow_summary_artifact_path", training_config.summary_artifact_path)
        mlflow.set_tag("mlflow_reports_artifact_dir", training_config.reports_artifact_dir)

        git_commit = get_git_commit_sha()
        if git_commit:
            # El commit permite reconstruir exactamente el estado del codigo
            # que produjo el entrenamiento. Es una pieza clave de trazabilidad.
            mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("config_path", str(args.config))

        # Los tags describen el contexto de ejecucion. Los parametros guardan
        # los hiperparametros y decisiones de configuracion que queremos poder
        # comparar entre runs.
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
            training_config.config_artifact_path,
        )

        # Antes de entrenar redes neuronales guardamos una referencia simple.
        # Usar la mediana del target como baseline permite medir si el modelo
        # realmente aprende algo mas util que una prediccion constante.
        baseline_value = float(np.median(y_train))
        baseline_predictions = np.full(shape=len(y_test), fill_value=baseline_value)
        baseline_metrics = evaluate_regression_metrics(y_test, baseline_predictions)
        mlflow.log_metrics({f"baseline_{key}": value for key, value in baseline_metrics.items()})

        experiment_results: list[dict[str, Any]] = []

        for experiment_config in experiment_configs:
            comparison_factory = create_model_factory(experiment_config.learning_rate)

            # Cada experimento se ejecuta como nested run. Esto hace que el run
            # padre concentre el pipeline completo, mientras que cada hijo
            # representa una familia de arquitecturas comparada de forma aislada.
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

                # La comparacion devuelve el mejor candidato del conjunto de
                # arquitecturas definidas para este experimento. Ademas de la
                # arquitectura seleccionada, conserva historia, metricas y el
                # modelo ya entrenado, para evitar reentrenar innecesariamente.
                best_result = compare_architectures_regression(
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

                # Se registran los resultados del mejor candidato del nested run.
                # Estos valores son los que luego se usan para comparar
                # experimentos entre si y para decidir el ganador global.
                mlflow.log_param("best_architecture", str(list(best_result.architecture)))
                mlflow.log_param("selection_criterion", best_result.criterion_order)
                mlflow.log_metric("best_valid_rmse", float(best_result.rmse_valid))
                mlflow.log_metric("best_valid_mae", float(best_result.mae_valid))
                mlflow.log_metric("best_valid_r2", float(best_result.r2_valid))
                mlflow.log_metric("best_val_loss", float(best_result.val_loss_minima))
                mlflow.log_metric("best_epochs_trained", float(best_result.epochs_trained))
                mlflow.log_metric("best_epoch", float(best_result.best_epoch))

                # A partir de este punto, reutilizamos el modelo seleccionado.
                # No se vuelve a entrenar: solo se evalua en test y se empaqueta
                # junto con el preprocesador para dejarlo listo para inferencia.
                selected_model = best_result.model
                selected_history = best_result.history

                x_test_final_processed = preprocessor.transform(x_test_clean)
                test_predictions = selected_model.predict(
                    x_test_final_processed,
                    verbose=0,
                ).ravel()
                test_metrics = evaluate_regression_metrics(y_test, test_predictions)
                mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

                # Guardar el historial completo resulta util para auditoria y
                # analisis posterior: permite ver la evolucion de loss y metricas
                # sin depender de la UI ni de la memoria del proceso.
                mlflow.log_dict(selected_history, training_config.history_artifact_path)

                # MLflow necesita artefactos serializados para poder reconstruir
                # el pipeline de inferencia. Por eso se persisten el preprocesador
                # y el modelo Keras en una carpeta temporal antes de loggearlos.
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    preprocessor_path, keras_model_path = dump_artifacts_to_tempdir(
                        preprocessor=preprocessor,
                        model=selected_model,
                        temp_dir=temp_dir,
                    )
                    input_example = x_test_raw.head(5)
                    signature = infer_signature(
                        input_example,
                        pd.DataFrame(
                            {
                                "prediction": selected_model.predict(
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
                        artifact_path=training_config.model_artifact_path,
                        python_model=pyfunc_model,
                        artifacts={
                            "preprocessor": str(preprocessor_path),
                            "keras_model": str(keras_model_path),
                        },
                        input_example=input_example,
                        signature=signature,
                    )

                    # El URI "runs:/..." apunta al modelo loggeado dentro de este
                    # mismo run. Es el identificador que luego se usa para
                    # registrar el artefacto final en el Model Registry.
                    model_uri = (
                        f"runs:/{mlflow.active_run().info.run_id}/"
                        f"{training_config.model_artifact_path}"
                    )
                    mlflow.log_param("model_uri", model_uri)

                experiment_results.append(
                    {
                        "experiment_name": experiment_config.name,
                        "run_id": mlflow.active_run().info.run_id,
                        "model_uri": model_uri,
                        "best_architecture": str(list(best_result.architecture)),
                        "selected_model": "best_validation_candidate",
                        "best_valid_rmse": float(best_result.rmse_valid),
                        "best_valid_mae": float(best_result.mae_valid),
                        "best_valid_r2": float(best_result.r2_valid),
                        "best_val_loss": float(best_result.val_loss_minima),
                        "best_epoch": int(best_result.best_epoch),
                        "test_rmse": float(test_metrics["rmse"]),
                        "test_mae": float(test_metrics["mae"]),
                        "test_r2": float(test_metrics["r2"]),
                        "criterion_order": experiment_config.criterion_order,
                }
                )

                mlflow.set_tag("status", "completed")

        # Al terminar todos los nested runs se consolida un resumen global con
        # todos los experimentos evaluados. Este resumen queda como artefacto y
        # tambien como JSON para consumo programatico.
        experiment_results_df = pd.DataFrame(experiment_results)
        summary_artifact = save_dataframe(
            experiment_results_df,
            Path(tempfile.gettempdir())
            / f"automobile_experiment_summary_{mlflow.active_run().info.run_id}.csv",
        )
        mlflow.log_artifact(
            str(summary_artifact),
            artifact_path=training_config.reports_artifact_dir,
        )
        mlflow.log_dict(
            experiment_results_df.to_dict(orient="records"),
            training_config.summary_artifact_path,
        )

        if experiment_results_df.empty:
            raise RuntimeError("No se pudo completar ningun experimento.")

        # El ganador global se define por la mejor validacion segun la metrica
        # principal. Aqui se usa RMSE de validacion, porque penaliza de forma mas
        # fuerte los errores grandes y es facil de interpretar en unidades del target.
        winner_row = experiment_results_df.sort_values(
            by="best_valid_rmse",
            ascending=True,
        ).iloc[0]

        mlflow.log_param("winning_experiment", winner_row["experiment_name"])
        mlflow.log_param("winning_model_uri", winner_row["model_uri"])
        mlflow.log_metric("winning_valid_rmse", float(winner_row["best_valid_rmse"]))
        mlflow.log_metric("winning_test_rmse", float(winner_row["test_rmse"]))

        # El registro en el Model Registry es opcional. Si falla, no abortamos el
        # entrenamiento: guardamos el error como tag para diagnostico posterior y
        # mantenemos el run completo disponible en MLflow.
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
