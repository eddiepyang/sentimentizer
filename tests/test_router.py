"""Tests for the SetFit router module.

Tests cover config, labels, seeds, dataset loading, and augmentation.
SetFit-dependent tests are skipped when the setfit package is not installed.
"""

import json
from pathlib import Path

import pytest

# Check if optional dependencies are available
try:
    import setfit  # noqa: F401

    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False

try:
    import datasets  # noqa: F401

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

skip_without_setfit = pytest.mark.skipif(
    not SETFIT_AVAILABLE,
    reason="setfit not installed (install with: pip install -e '.[router]')",
)

skip_without_datasets = pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="datasets not installed (install with: pip install -e '.[router]')",
)


class TestSetFitConfig:
    """Test SetFitConfig defaults and immutability."""

    def test_default_config(self) -> None:
        from sentimentizer.router.config import SetFitConfig

        config = SetFitConfig()
        assert config.base_model == "BAAI/bge-base-en-v1.5"
        assert config.num_iterations == 20
        assert config.num_epochs == 1
        assert config.batch_size == 16
        assert config.max_seq_length == 512
        assert config.seed == 42
        assert config.output_dir == Path("models/router")

    def test_frozen_config(self) -> None:
        from sentimentizer.router.config import SetFitConfig

        config = SetFitConfig()
        with pytest.raises(AttributeError):
            config.base_model = "different-model"  # type: ignore[misc]

    def test_custom_config(self) -> None:
        from sentimentizer.router.config import SetFitConfig

        config = SetFitConfig(
            base_model="mxbai-embed-large-v1",
            num_iterations=10,
            batch_size=32,
        )
        assert config.base_model == "mxbai-embed-large-v1"
        assert config.num_iterations == 10
        assert config.batch_size == 32


class TestRouteLabels:
    """Test RouteLabels config and label mappings."""

    def test_default_labels(self) -> None:
        from sentimentizer.router.config import RouteLabels

        labels = RouteLabels()
        assert labels.dietary == 0
        assert labels.service == 1
        assert labels.general == 2

    def test_label_names(self) -> None:
        from sentimentizer.router.config import RouteLabels

        names = RouteLabels.label_names()
        assert names == {0: "dietary", 1: "service", 2: "general"}

    def test_num_classes(self) -> None:
        from sentimentizer.router.config import RouteLabels

        assert RouteLabels.num_classes() == 3


class TestSeedUtterances:
    """Test seed utterance data integrity."""

    def test_seed_utterances_exist(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        assert len(SEED_UTTERANCES) > 0

    def test_seed_utterances_have_correct_labels(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        valid_labels = {0, 1, 2}
        for entry in SEED_UTTERANCES:
            assert entry["label"] in valid_labels, f"Invalid label: {entry['label']}"

    def test_seed_utterances_have_nonempty_text(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        for entry in SEED_UTTERANCES:
            assert len(entry["text"]) > 0, "Empty text found in seed utterances"

    def test_utterances_per_category(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        counts = {0: 0, 1: 0, 2: 0}
        for entry in SEED_UTTERANCES:
            counts[entry["label"]] += 1

        # Each category should have 10 seeds
        assert counts[0] == 10, f"Dietary has {counts[0]} seeds, expected 10"
        assert counts[1] == 10, f"Service has {counts[1]} seeds, expected 10"
        assert counts[2] == 10, f"General has {counts[2]} seeds, expected 10"

    def test_total_seed_count(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        assert len(SEED_UTTERANCES) == 30


@skip_without_datasets
class TestDatasetLoader:
    """Test JSONL dataset loading (with mock data)."""

    def test_load_router_dataset(self, tmp_path) -> None:
        """Test loading and splitting a JSONL dataset."""
        from sentimentizer.router.dataset import load_router_dataset

        # Create a mock JSONL file
        data = []
        for i in range(100):
            label = i % 3
            data.append({"text": f"Review text {i} for label {label}", "label": label})

        jsonl_path = tmp_path / "test_data.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        train_ds, test_ds = load_router_dataset(str(jsonl_path), test_size=0.2, seed=42)

        # Should have 80% train, 20% test
        assert len(train_ds) == 80
        assert len(test_ds) == 20

    def test_load_router_dataset_preserves_columns(self, tmp_path) -> None:
        """Test that loaded dataset has 'text' and 'label' columns."""
        from sentimentizer.router.dataset import load_router_dataset

        data = [
            {"text": "Dietary review", "label": 0},
            {"text": "Service review", "label": 1},
            {"text": "General review", "label": 2},
        ] * 10  # Need enough for train/test split

        jsonl_path = tmp_path / "test_columns.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        train_ds, test_ds = load_router_dataset(str(jsonl_path), test_size=0.3, seed=42)

        assert "text" in train_ds.column_names
        assert "label" in train_ds.column_names


class TestSaveDatasetToJsonl:
    """Test saving utterance dicts to JSONL format."""

    def test_save_dataset_to_jsonl(self, tmp_path) -> None:
        from sentimentizer.router.dataset import save_dataset_to_jsonl

        data = [
            {"text": "Review 1", "label": 0},
            {"text": "Review 2", "label": 1},
        ]

        output_path = tmp_path / "output.jsonl"
        result = save_dataset_to_jsonl(data, str(output_path))

        assert result.exists()
        with open(result) as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Verify JSONL format
        entry = json.loads(lines[0])
        assert "text" in entry
        assert "label" in entry


class TestAugmentConfig:
    """Test AugmentConfig and load_router_config."""

    def test_augment_config_defaults(self) -> None:
        from sentimentizer.router.config import AugmentConfig

        config = AugmentConfig()
        assert config.model == "glm-5.1:cloud"
        assert config.ollama_url == "http://localhost:11434/api/generate"
        assert config.variations_per_seed == 50
        assert config.output_path == "augmented_yelp.jsonl"
        assert config.batch_size == 5

    def test_augment_config_custom(self) -> None:
        from sentimentizer.router.config import AugmentConfig

        config = AugmentConfig(model="llama3", variations_per_seed=10, output_path="custom.jsonl")
        assert config.model == "llama3"
        assert config.variations_per_seed == 10
        assert config.output_path == "custom.jsonl"

    def test_load_router_config_defaults(self) -> None:
        from sentimentizer.router.config import load_router_config

        train_cfg, augment_cfg = load_router_config()
        assert train_cfg.base_model == "BAAI/bge-base-en-v1.5"
        assert train_cfg.num_iterations == 20
        assert augment_cfg.model == "glm-5.1:cloud"
        assert augment_cfg.variations_per_seed == 50

    def test_load_router_config_custom_file(self, tmp_path) -> None:
        from sentimentizer.router.config import load_router_config

        config_content = """
training:
  base_model: "custom-model"
  num_iterations: 30

augmentation:
  model: "custom-llm"
  variations_per_seed: 100
"""
        config_path = tmp_path / "custom_router.yaml"
        config_path.write_text(config_content)

        train_cfg, augment_cfg = load_router_config(str(config_path))
        assert train_cfg.base_model == "custom-model"
        assert train_cfg.num_iterations == 30
        assert augment_cfg.model == "custom-llm"
        assert augment_cfg.variations_per_seed == 100

    def test_load_router_config_missing_file(self) -> None:
        import pytest

        from sentimentizer.router.config import load_router_config

        with pytest.raises(FileNotFoundError):
            load_router_config("/nonexistent/path.yaml")


class TestAugmentSeeds:
    """Test seed augmentation (with mocked Ollama API)."""

    def test_build_prompt(self) -> None:
        from sentimentizer.router.augment import _build_prompt

        prompt = _build_prompt("Great food!", 2, 10)
        assert "Great food!" in prompt
        assert "general" in prompt.lower()
        assert "10" in prompt

    def test_augment_seeds_handles_api_failure(self) -> None:
        """Augment seeds should gracefully handle API failures."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        # Use a URL that doesn't exist — should return just the original seeds
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
        )
        # Should still return the original seeds even if API fails
        assert len(result) >= len(seeds)
        assert result[0] == seeds[0]

    def test_augment_seeds_streams_to_file(self, tmp_path) -> None:
        """Augment seeds with output_path should stream entries to JSONL."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        output_path = tmp_path / "streamed.jsonl"

        augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
        )

        # File should exist and contain seed utterances
        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        # At minimum, the seed entries should be written
        assert len(lines) >= len(seeds)
        entry = json.loads(lines[0])
        assert "text" in entry
        assert "label" in entry

    def test_augment_seeds_without_output_path(self) -> None:
        """Augment seeds without output_path should still return results."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
        )
        assert len(result) >= len(seeds)
        assert result[0] == seeds[0]

    def test_augment_seeds_resume_skips_existing(self, tmp_path) -> None:
        """Resume mode should skip seeds already in the output file."""
        from sentimentizer.router.augment import augment_seeds

        # Create an existing file with some entries
        output_path = tmp_path / "resume.jsonl"
        existing_entries = [
            {"text": "Seed 1", "label": 0},
            {"text": "Seed 2", "label": 1},
            {"text": "Augmented 1", "label": 0},
        ]
        with open(output_path, "w") as f:
            for entry in existing_entries:
                f.write(json.dumps(entry) + "\n")

        # Resume with seeds that include the existing ones
        seeds = [
            {"text": "Seed 1", "label": 0},  # already in file — should be skipped
            {"text": "Seed 2", "label": 1},  # already in file — should be skipped
            {"text": "Seed 3", "label": 2},  # NOT in file — should be processed
        ]

        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
            resume=True,
        )

        # Should include the 3 existing entries + seed 3 (API will fail but seed is kept)
        assert len(result) >= 3  # at least the existing entries

        # File should still have the original entries
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) >= 3  # at least the original entries

    def test_augment_seeds_resume_all_processed(self, tmp_path) -> None:
        """Resume mode with all seeds already processed should return existing."""
        from sentimentizer.router.augment import augment_seeds

        output_path = tmp_path / "complete.jsonl"
        seeds = [{"text": "Seed A", "label": 0}, {"text": "Seed B", "label": 1}]

        # Write all seeds to file
        with open(output_path, "w") as f:
            for seed in seeds:
                f.write(json.dumps(seed) + "\n")

        # Resume should detect all seeds are already processed
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
            resume=True,
        )

        # Should return the existing entries without making API calls
        assert len(result) == 2
        assert result[0]["text"] == "Seed A"
        assert result[1]["text"] == "Seed B"
