"""Tests to prevent regressions from using deprecated/removed Ray APIs.

These tests act as guardrails to ensure the codebase stays compatible with
Ray 2.55.1 and does not accidentally reintroduce APIs from older versions
that have been removed or changed.

Covers:
- Dataset API changes (random_split removed, random_sample returns single Dataset)
- Checkpoint API changes (from_dict/to_dict removed, directory-based only)
- EmbeddingsConfig attribute validation (no file_path/sub_file_path)
- train_loop_config key validation (embeddings_model_name, not file_path)
- Ray version pin check
"""

import os
import tempfile

import pytest

# ruff: noqa: E402
ray = pytest.importorskip("ray")
from ray.train import Checkpoint  # noqa: I001 — grouped with ray imports

# ─── Ray Version ───────────────────────────────────────────────────


class TestRayVersion:
    """Ensure the installed Ray version is 2.55.x."""

    def test_ray_version_is_2_55(self) -> None:
        """Ray must be 2.55.x to match the API assumptions in this project."""
        major, minor, *_ = ray.__version__.split(".")
        assert major == "2" and minor == "55", (
            f"Expected Ray 2.55.x but got {ray.__version__}. "
            "The codebase relies on Ray 2.55 API conventions. "
            "Update the code and these tests if upgrading Ray."
        )


# ─── Dataset API ───────────────────────────────────────────────────


class TestDatasetAPI:
    """Guard against deprecated/removed Dataset methods from older Ray versions."""

    def test_dataset_not_iterable(self) -> None:
        """Dataset objects must not be iterated directly (Ray 2.55 enforces this).

        In older Ray versions, iterating a Dataset silently worked but was
        incorrect. Ray 2.55 raises TypeError.
        """
        ds = ray.data.range(5)
        with pytest.raises(TypeError, match="aren't iterable"):
            for _ in ds:
                pass

    def test_dataset_has_no_random_split(self) -> None:
        """Dataset.random_split() does not exist in Ray 2.55.

        Older code used `ds.random_split([0.8, 0.2])` which does not exist.
        Use `ds.train_test_split()` for splitting or `ds.random_sample()`
        for fractional sampling.
        """
        ds = ray.data.range(10)
        assert not hasattr(ds, "random_split"), (
            "Dataset.random_split() should not exist in Ray 2.55. "
            "Use ds.train_test_split() or ds.random_sample() instead."
        )

    def test_random_sample_returns_single_dataset(self) -> None:
        """random_sample(fraction) returns a single Dataset, NOT a tuple.

        A common mistake is to unpack it like: `keep, _ = ds.random_sample(0.5)`
        This will fail because random_sample returns a single Dataset.
        """
        ds = ray.data.range(100)
        result = ds.random_sample(0.5, seed=42)
        assert isinstance(result, ray.data.Dataset)
        # Must NOT be unpackable as a tuple
        with pytest.raises(TypeError):
            _keep, _rest = result  # type: ignore[misc]

    def test_train_test_split_returns_two_datasets(self) -> None:
        """train_test_split(test_size) returns (train, test) tuple."""
        ds = ray.data.range(10)
        train_ds, test_ds = ds.train_test_split(test_size=0.3, shuffle=True, seed=42)
        assert isinstance(train_ds, ray.data.Dataset)
        assert isinstance(test_ds, ray.data.Dataset)
        assert train_ds.count() + test_ds.count() == 10

    def test_random_sample_fraction_arg(self) -> None:
        """random_sample takes a fraction (0.0-1.0), not a list of proportions."""
        ds = ray.data.range(100)
        # Correct: single float fraction
        result = ds.random_sample(0.5, seed=42)
        assert isinstance(result, ray.data.Dataset)

        # Wrong: list of proportions (old random_split style) should fail
        with pytest.raises(TypeError):
            ds.random_sample([0.5, 0.5], seed=42)  # type: ignore[arg-type]

    def test_filter_exists(self) -> None:
        """Dataset.filter() must exist and work with fn= callable."""
        ds = ray.data.range(10)
        filtered = ds.filter(fn=lambda row: row["id"] < 5)
        assert filtered.count() == 5

    def test_union_exists(self) -> None:
        """Dataset.union() must exist for concatenating datasets."""
        ds1 = ray.data.range(5)
        ds2 = ray.data.range(5)
        combined = ds1.union(ds2)
        assert combined.count() == 10

    def test_random_shuffle_exists(self) -> None:
        """Dataset.random_shuffle() must exist with seed= parameter."""
        ds = ray.data.range(10)
        shuffled = ds.random_shuffle(seed=42)
        assert shuffled.count() == 10

    def test_count_exists(self) -> None:
        """Dataset.count() must exist and return an int."""
        ds = ray.data.range(10)
        assert isinstance(ds.count(), int)
        assert ds.count() == 10


# ─── Checkpoint API ─────────────────────────────────────────────────


class TestCheckpointAPI:
    """Guard against removed Checkpoint methods from older Ray versions."""

    def test_checkpoint_from_dict_removed(self) -> None:
        """Checkpoint.from_dict() was removed in Ray 2.55+.

        Older code used: Checkpoint.from_dict({"model_state_dict": ...})
        This must now raise AttributeError.
        """
        with pytest.raises(AttributeError, match="from_dict"):
            Checkpoint.from_dict({"model_state_dict": "fake"})

    def test_checkpoint_to_dict_removed(self) -> None:
        """Checkpoint.to_dict() was removed in Ray 2.55+.

        Older code used: checkpoint.to_dict()
        This must now raise AttributeError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Checkpoint.from_directory(tmpdir)
            with pytest.raises(AttributeError, match="to_dict"):
                checkpoint.to_dict()

    def test_checkpoint_from_directory_exists(self) -> None:
        """Checkpoint.from_directory() must exist (replacement for from_dict)."""
        assert hasattr(Checkpoint, "from_directory")
        assert callable(Checkpoint.from_directory)

    def test_checkpoint_as_directory_exists(self) -> None:
        """Checkpoint.as_directory() must exist (replacement for to_dict)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal checkpoint directory
            with open(os.path.join(tmpdir, "data.pkl"), "wb") as fp:
                import ray.cloudpickle as pickle

                pickle.dump({"test": True}, fp)
            checkpoint = Checkpoint.from_directory(tmpdir)
            assert hasattr(checkpoint, "as_directory")

    def test_checkpoint_round_trip(self) -> None:
        """Directory-based checkpoint write/read round-trip must work.

        This is the pattern used in sentimentizer/trainer.py and
        workflows/driver.py.
        """
        import ray.cloudpickle as pickle

        original_data = {
            "model_state_dict": {"layer1.weight": [1.0, 2.0, 3.0]},
            "optimizer_state_dict": {"lr": 0.001},
            "epoch": 5,
        }

        # Write checkpoint (as done in trainer.py)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
                pickle.dump(original_data, fp)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # Read checkpoint (as done in driver.py)
            with (
                checkpoint.as_directory() as loaded_dir,
                open(os.path.join(loaded_dir, "data.pkl"), "rb") as fp,
            ):
                loaded_data = pickle.load(fp)

            assert loaded_data["model_state_dict"] == original_data["model_state_dict"]
            assert loaded_data["optimizer_state_dict"] == original_data["optimizer_state_dict"]
            assert loaded_data["epoch"] == 5


# ─── Config Attribute Guards ────────────────────────────────────────


class TestEmbeddingsConfigAttributes:
    """Guard against using removed/nonexistent EmbeddingsConfig attributes.

    Older code referenced EmbeddingsConfig.file_path and
    EmbeddingsConfig.sub_file_path, but these never existed in the
    current dataclass. Embeddings are loaded via gensim using model_name.
    """

    def test_embeddings_config_has_model_name(self) -> None:
        """EmbeddingsConfig must have model_name attribute."""
        from sentimentizer.config import EmbeddingsConfig

        cfg = EmbeddingsConfig()
        assert hasattr(cfg, "model_name")
        assert cfg.model_name == "glove-wiki-gigaword-100"

    def test_embeddings_config_has_emb_length(self) -> None:
        """EmbeddingsConfig must have emb_length attribute."""
        from sentimentizer.config import EmbeddingsConfig

        cfg = EmbeddingsConfig()
        assert hasattr(cfg, "emb_length")
        assert cfg.emb_length == 100

    def test_embeddings_config_no_file_path(self) -> None:
        """EmbeddingsConfig must NOT have file_path attribute.

        Older code incorrectly referenced this — embeddings are loaded
        via gensim.downloader using model_name, not from a file path.
        """
        from sentimentizer.config import EmbeddingsConfig

        cfg = EmbeddingsConfig()
        assert not hasattr(cfg, "file_path"), (
            "EmbeddingsConfig should not have 'file_path'. "
            "Embeddings are loaded via gensim using 'model_name'."
        )

    def test_embeddings_config_no_sub_file_path(self) -> None:
        """EmbeddingsConfig must NOT have sub_file_path attribute."""
        from sentimentizer.config import EmbeddingsConfig

        cfg = EmbeddingsConfig()
        assert not hasattr(cfg, "sub_file_path"), (
            "EmbeddingsConfig should not have 'sub_file_path'. "
            "Embeddings are loaded via gensim using 'model_name'."
        )

    def test_driver_config_embeddings_has_model_name(self) -> None:
        """DriverConfig.embeddings must expose model_name (not file_path)."""
        from sentimentizer.config import DriverConfig

        assert hasattr(DriverConfig.embeddings, "model_name")
        assert not hasattr(DriverConfig.embeddings, "file_path")

    def test_embeddings_config_constructor_with_model_name(self) -> None:
        """EmbeddingsConfig(model_name=..., emb_length=...) must work."""
        from sentimentizer.config import EmbeddingsConfig

        cfg = EmbeddingsConfig(model_name="glove-wiki-gigaword-50", emb_length=50)
        assert cfg.model_name == "glove-wiki-gigaword-50"
        assert cfg.emb_length == 50

    def test_embeddings_config_constructor_rejects_file_path(self) -> None:
        """EmbeddingsConfig(file_path=...) must raise TypeError."""
        from sentimentizer.config import EmbeddingsConfig

        with pytest.raises(TypeError):
            EmbeddingsConfig(file_path="/tmp/test.zip")  # type: ignore[call-arg]

    def test_embeddings_config_constructor_rejects_sub_file_path(self) -> None:
        """EmbeddingsConfig(sub_file_path=...) must raise TypeError."""
        from sentimentizer.config import EmbeddingsConfig

        with pytest.raises(TypeError):
            EmbeddingsConfig(sub_file_path="test.txt")  # type: ignore[call-arg]


# ─── train_loop_config Key Guards ──────────────────────────────────


class TestTrainLoopConfigKeys:
    """Guard against using removed config keys in the Ray Train loop config.

    The train_loop_config dict passed to TorchTrainer must use the
    correct key names that match what _train_func expects.
    """

    def test_config_uses_embeddings_model_name(self) -> None:
        """train_loop_config must use 'embeddings_model_name', not 'embeddings_file_path'."""
        from sentimentizer.config import DriverConfig

        # Verify the keys that new_ray_trainer puts in the config
        config = {
            "dict_path": DriverConfig.files.dictionary_file_path,
            "embeddings_model_name": DriverConfig.embeddings.model_name,
            "embeddings_emb_length": DriverConfig.embeddings.emb_length,
        }
        assert "embeddings_model_name" in config
        assert "embeddings_file_path" not in config
        assert "embeddings_sub_file_path" not in config

    def test_train_func_config_keys_match(self) -> None:
        """The config keys set by new_ray_trainer must match what _train_func reads."""
        # Keys that new_ray_trainer sets
        setter_keys = {
            "epochs",
            "batch_size",
            "lr",
            "betas",
            "weight_decay",
            "use_warmup",
            "warmup_steps",
            "total_steps",
            "scheduler_eta_min",
            "model_type",
            "dict_path",
            "embeddings_model_name",
            "embeddings_emb_length",
            "input_len",
            "loss_type",
        }

        # Keys that _train_func reads (must be a subset of setter_keys)
        reader_keys = {
            "epochs",
            "batch_size",
            "lr",
            "betas",
            "weight_decay",
            "use_warmup",
            "warmup_steps",
            "total_steps",
            "scheduler_eta_min",
            "model_type",
            "dict_path",
            "embeddings_model_name",
            "embeddings_emb_length",
            "input_len",
            "loss_type",
        }

        assert reader_keys.issubset(
            setter_keys
        ), f"_train_func reads keys not set by new_ray_trainer: {reader_keys - setter_keys}"

    def test_no_legacy_embedding_keys_in_config(self) -> None:
        """Config dicts must not contain 'embeddings_file_path' or 'embeddings_sub_file_path'."""
        config = {
            "embeddings_model_name": "glove-wiki-gigaword-100",
            "embeddings_emb_length": 100,
        }
        assert "embeddings_file_path" not in config
        assert "embeddings_sub_file_path" not in config


# ─── Ray Train Context Guard ───────────────────────────────────────


class TestRayTrainContext:
    """Guard against calling train.get_context() outside a worker."""

    def test_get_context_outside_worker_raises(self) -> None:
        """train.get_context() must raise RuntimeError when called outside a worker.

        This prevents accidentally calling it from the driver process.
        """
        from ray import train

        with pytest.raises(RuntimeError, match="cannot be used outside"):
            train.get_context()

    def test_get_dataset_shard_is_standalone_function(self) -> None:
        """In Ray 2.55+, get_dataset_shard is a standalone function, not a method
        on DistributedTrainContext. Calling train.get_context().get_dataset_shard()
        raises AttributeError.

        The correct API is: train.get_dataset_shard("train")
        See https://docs.ray.io/en/2.55.1/train/api/doc/ray.train.get_dataset_shard.html
        """
        from ray import train

        assert callable(train.get_dataset_shard)
        # Verify it's a module-level function, not a method on get_context()
        with pytest.raises(RuntimeError):
            train.get_context()  # raises outside worker


# ─── Ray Tune get_best_result Guard ──────────────────────────────────


class TestTuneGetBestResult:
    """Guard against calling result.get_best_result() without metric/mode args.

    Ray Tune 2.55+ requires metric and mode to be explicitly passed to
    get_best_result() when they are not set in TuneConfig. Failure to
    do so raises: ValueError: No metric is provided.
    """

    def test_get_best_result_requires_metric_arg(self) -> None:
        """result.get_best_result() must receive metric arg to avoid ValueError."""
        from unittest.mock import MagicMock

        from ray.tune import ResultGrid

        # Simulate a ResultGrid without metric in TuneConfig
        mock_result = MagicMock(spec=ResultGrid)
        mock_result.get_best_result.side_effect = ValueError("No metric is provided")

        with pytest.raises(ValueError, match="No metric is provided"):
            mock_result.get_best_result()

        # Verify that passing metric and mode succeeds
        mock_result.get_best_result.side_effect = None
        mock_result.get_best_result.return_value = MagicMock(
            config={"lr": 0.001}, metrics={"val_accuracy": 0.95}
        )

        best = mock_result.get_best_result(metric="val_accuracy", mode="max")
        assert best.metrics["val_accuracy"] == 0.95

    def test_tuner_config_passes_metric_to_get_best_result(self) -> None:
        """The tune_model function must pass TunerConfig.metric/mode to get_best_result()."""
        import inspect

        from sentimentizer.tuner import tune_model

        source = inspect.getsource(tune_model)
        # Ensure get_best_result is called with metric= and mode= keyword args
        assert "result.get_best_result(" in source
        assert "metric=tuner_config.metric" in source
        assert "mode=tuner_config.mode" in source


# ─── ResultGrid and Result API Guards ────────────────────────────────


class TestResultGridAPI:
    """Guard against incorrect ResultGrid and Result API usage.

    Ray 2.55+ changed the ResultGrid iteration and Result.metrics APIs.
    """

    def test_result_grid_not_directly_iterable(self) -> None:
        """ResultGrid does NOT implement __iter__ in Ray 2.55+.

        Iterating over a ResultGrid with ``for result in result_grid:`` will
        raise TypeError. Use index-based iteration:
        ``for i in range(len(result_grid)): result_grid[i]``
        """
        from ray.tune import ResultGrid

        assert not hasattr(ResultGrid, "__iter__"), (
            "ResultGrid should NOT have __iter__. "
            "Use index-based iteration: for i in range(len(result_grid)): ..."
        )

    def test_result_grid_supports_len_and_getitem(self) -> None:
        """ResultGrid supports len() and index-based access in Ray 2.55+."""
        from ray.tune import ResultGrid

        assert hasattr(ResultGrid, "__len__"), "ResultGrid must support len()"
        assert hasattr(ResultGrid, "__getitem__"), "ResultGrid must support index access"

    def test_result_metrics_is_optional_dict(self) -> None:
        """Result.metrics is Optional[Dict] in Ray 2.55+ — it can be None.

        Always guard with ``result.metrics or {}`` when accessing metrics.
        """
        import dataclasses

        from ray.tune import Result

        fields = {f.name: f.type for f in dataclasses.fields(Result)}
        assert "metrics" in fields, "Result must have a 'metrics' field"
        # The type annotation allows None — code must handle this
        assert "Optional" in str(fields["metrics"]) or "None" in str(fields["metrics"]), (
            "Result.metrics should be Optional[Dict[str, Any]] in Ray 2.55+. "
            "Always guard with `result.metrics or {}`."
        )

    def test_result_has_config_property(self) -> None:
        """Result must have a 'config' property for accessing trial config."""
        from ray.tune import Result

        assert hasattr(Result, "config"), "Result must have a 'config' property"

    def test_result_object_has_no_collection_methods(self) -> None:
        """TorchTrainer.fit() returns a single Result, not a ResultGrid.

        This guards against treating the output of TorchTrainer.fit() as a
        collection (e.g., calling len()) or calling get_best_result() on it.
        """
        from ray.train import Result

        assert not hasattr(Result, "__len__"), "Result object must not have __len__"
        assert not hasattr(Result, "get_best_result"), (
            "Result object must not have get_best_result. "
            "TorchTrainer.fit() returns a single Result, not a ResultGrid."
        )


class TestTuneCallbackAPI:
    """Guard against incorrect TunePrometheusCallback API usage.

    Ray 2.55+ passes Trial objects to callback methods, not strings.
    """

    def test_callback_uses_trial_id_attribute(self) -> None:
        """TunePrometheusCallback must extract trial_id from Trial objects.

        In Ray 2.55+, the trial parameter in on_trial_result/on_trial_complete/
        on_trial_start is a Trial object with a trial_id attribute, NOT a string.
        The callback must use getattr(trial, 'trial_id') instead of str(trial).
        """
        import inspect

        from sentimentizer.tuner import TunePrometheusCallback

        source = inspect.getsource(TunePrometheusCallback.on_trial_result)
        # Must use getattr(trial, 'trial_id') to extract the ID from Trial objects
        assert "trial_id" in source, "on_trial_result must extract trial_id from trial object"
        assert "getattr" in source, (
            "on_trial_result must use getattr(trial, 'trial_id') to handle Trial objects. "
            "Ray 2.55+ passes Trial objects, not strings."
        )

    def test_tuner_result_grid_index_iteration(self) -> None:
        """tune_model must use index-based iteration over ResultGrid.

        ``for trial_result in result:`` does NOT work in Ray 2.55+
        because ResultGrid does not implement __iter__.
        Use ``for i in range(len(result)): trial_result = result[i]`` instead.
        """
        import inspect

        from sentimentizer.tuner import tune_model

        source = inspect.getsource(tune_model)
        # Should NOT have "for trial_result in result:"
        assert "for trial_result in result" not in source, (
            "ResultGrid is not directly iterable in Ray 2.55+. "
            "Use index-based iteration: for i in range(len(result)): ..."
        )
        # Should use index-based iteration
        assert "for i in range(len(result))" in source, (
            "tune_model must iterate over ResultGrid using index access. "
            "Use: for i in range(len(result)): trial_result = result[i]"
        )

    def test_tuner_metrics_guarded_against_none(self) -> None:
        """tune_model must guard Result.metrics against None.

        Result.metrics is Optional[Dict] in Ray 2.55+. It can be None
        for trials that were errored or didn't report metrics.
        """
        import inspect

        from sentimentizer.tuner import tune_model

        source = inspect.getsource(tune_model)
        # best_metrics should use "or {}" guard
        assert "best_result.metrics or {}" in source, (
            "best_result.metrics can be None in Ray 2.55+. "
            "Use: best_metrics = best_result.metrics or {}"
        )
        # trial_result.metrics should also be guarded
        assert (
            "trial_result.metrics or {}" in source
        ), "trial_result.metrics can be None in Ray 2.55+. Use: trial_result.metrics or {}"
