from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from src.rnn_price_predict import BestArchitectureResult


@dataclass
class _FakeRun:
    run_id: str

    @property
    def info(self) -> SimpleNamespace:
        return SimpleNamespace(run_id=self.run_id)


class _FakeRunManager:
    def __init__(self, mlflow_module: ModuleType, run_name: str | None = None, nested: bool = False) -> None:
        self._mlflow_module = mlflow_module
        self._run_name = run_name
        self._nested = nested
        self._run: _FakeRun | None = None

    def __enter__(self) -> _FakeRun:
        run_id = f"run_{len(self._mlflow_module._runs) + 1}"
        self._run = _FakeRun(run_id)
        self._mlflow_module._stack.append(self._run)
        self._mlflow_module._runs.append(
            {
                "run_id": run_id,
                "run_name": self._run_name,
                "nested": self._nested,
            }
        )
        return self._run

    def __exit__(self, exc_type, exc, tb) -> None:
        self._mlflow_module._stack.pop()


class _DummySelectedModel:
    def __init__(self) -> None:
        self.fit_called = False
        self.save_calls: list[Path] = []
        self.predict_calls = 0

    def fit(self, *args, **kwargs):  # pragma: no cover - guard against regressions
        self.fit_called = True
        raise AssertionError("El modelo seleccionado no debe reentrenarse.")

    def predict(self, X, verbose: int = 0):
        self.predict_calls += 1
        return np.zeros((len(X), 1), dtype=float)

    def save(self, output_path):
        path = Path(output_path)
        path.write_bytes(b"dummy-model")
        self.save_calls.append(path)


def _build_fake_mlflow_module() -> ModuleType:
    fake_mlflow = ModuleType("mlflow")
    fake_mlflow.__path__ = []  # type: ignore[attr-defined]
    fake_mlflow._stack = deque()
    fake_mlflow._runs = []
    fake_mlflow.logged_params = []
    fake_mlflow.logged_metrics = []
    fake_mlflow.logged_dicts = []
    fake_mlflow.logged_artifacts = []
    fake_mlflow.logged_tags = []
    fake_mlflow.registered_models = []

    def active_run():
        return fake_mlflow._stack[-1] if fake_mlflow._stack else None

    def set_tracking_uri(uri):
        fake_mlflow.tracking_uri = uri

    def set_experiment(name):
        fake_mlflow.experiment_name = name

    def set_tag(key, value):
        fake_mlflow.logged_tags.append((key, value))

    def log_params(params):
        fake_mlflow.logged_params.append(params)

    def log_param(key, value):
        fake_mlflow.logged_params.append({key: value})

    def log_metrics(metrics):
        fake_mlflow.logged_metrics.append(metrics)

    def log_metric(key, value):
        fake_mlflow.logged_metrics.append({key: value})

    def log_dict(data, artifact_file):
        fake_mlflow.logged_dicts.append((data, artifact_file))

    def log_artifact(path, artifact_path=None):
        fake_mlflow.logged_artifacts.append((path, artifact_path))

    def register_model(model_uri, registered_model_name):
        fake_mlflow.registered_models.append((model_uri, registered_model_name))
        return {"model_uri": model_uri, "registered_model_name": registered_model_name}

    def start_run(run_name=None, nested=False):
        return _FakeRunManager(fake_mlflow, run_name=run_name, nested=nested)

    fake_mlflow.active_run = active_run
    fake_mlflow.set_tracking_uri = set_tracking_uri
    fake_mlflow.set_experiment = set_experiment
    fake_mlflow.set_tag = set_tag
    fake_mlflow.log_params = log_params
    fake_mlflow.log_param = log_param
    fake_mlflow.log_metrics = log_metrics
    fake_mlflow.log_metric = log_metric
    fake_mlflow.log_dict = log_dict
    fake_mlflow.log_artifact = log_artifact
    fake_mlflow.register_model = register_model
    fake_mlflow.start_run = start_run

    fake_pyfunc = ModuleType("mlflow.pyfunc")

    class PythonModel:
        pass

    def log_model(*args, **kwargs):
        fake_pyfunc.logged_model = {"args": args, "kwargs": kwargs}

    fake_pyfunc.PythonModel = PythonModel
    fake_pyfunc.log_model = log_model

    fake_models = ModuleType("mlflow.models")
    fake_models.__path__ = []  # type: ignore[attr-defined]
    fake_signature = ModuleType("mlflow.models.signature")

    def infer_signature(inputs, outputs):
        return {"inputs": inputs, "outputs": outputs}

    fake_signature.infer_signature = infer_signature
    fake_models.signature = fake_signature
    fake_mlflow.pyfunc = fake_pyfunc
    fake_mlflow.models = fake_models

    return fake_mlflow


@pytest.mark.integration
def test_run_training_pipeline_uses_selected_model_without_retraining(
    monkeypatch,
    vehicle_data,
) -> None:
    from src import train as train_module

    fake_mlflow = _build_fake_mlflow_module()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_mlflow.pyfunc)
    monkeypatch.setitem(sys.modules, "mlflow.models", fake_mlflow.models)
    monkeypatch.setitem(sys.modules, "mlflow.models.signature", fake_mlflow.models.signature)

    selected_model = _DummySelectedModel()
    best_result = BestArchitectureResult(
        ranking=1,
        architecture=(32, 16),
        history={"loss": [1.2], "val_loss": [0.8]},
        model=selected_model,
        rmse_valid=1.0,
        mae_valid=0.9,
        r2_valid=0.8,
        val_loss_minima=0.8,
        epochs_trained=1,
        best_epoch=1,
        criterion_order="rmse_valid",
    )

    monkeypatch.setattr(train_module, "load_yaml_config", lambda path: {})
    monkeypatch.setattr(train_module, "load_dataset", lambda path, sep=",": vehicle_data.copy())
    monkeypatch.setattr(train_module, "compare_architectures_regression", lambda **kwargs: best_result)

    args = SimpleNamespace(
        config=Path("config/train.yaml"),
        data_path=Path("data/input/vehicle_data.csv"),
        tracking_uri=None,
        tracking_dir=Path("mlruns"),
        experiment_name="automobile-price-prediction",
        registered_model_name="automobile-price-predictor",
        model_artifact_path="model",
        config_artifact_path="config/config.json",
        history_artifact_path="reports/final_training_history.json",
        summary_artifact_path="reports/experiment_summary.json",
        reports_artifact_dir="reports",
        epochs=5,
        batch_size=None,
        patience=2,
        learning_rate=0.001,
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        run_name="test-run",
        no_registration=True,
        verbose_training=0,
        use_early_stopping=True,
    )

    train_module.run_training_pipeline(args)

    assert selected_model.fit_called is False
    assert selected_model.predict_calls >= 1
    assert fake_mlflow._runs
    assert fake_mlflow.pyfunc.logged_model["kwargs"]["artifact_path"] == "model"
    assert fake_mlflow.logged_dicts[0][1] == "config/config.json"
    assert fake_mlflow.logged_dicts[-1][1] == "reports/experiment_summary.json"
    assert fake_mlflow.logged_artifacts[-1][1] == "reports"
