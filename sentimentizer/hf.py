from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, HF_WEIGHTS_REPOS

logger = new_logger(DEFAULT_LOG_LEVEL)


# ---------------------------------------------------------------------------
# Model descriptions per model type
# ---------------------------------------------------------------------------

_MODEL_DESCRIPTIONS: dict[str, str] = {
    "rnn": (
        "A bidirectional LSTM for sentiment classification built on pre-trained GloVe "
        "embeddings. The model processes token sequences through a multi-layer "
        "bidirectional LSTM, concatenates the final forward and backward hidden "
        "states, and classifies via a two-layer MLP head."
    ),
    "encoder": (
        "A Transformer Encoder for sentiment classification built on pre-trained GloVe "
        "embeddings. The model uses multi-head self-attention with positional "
        "encodings and a classification token (CLS) to produce a sentiment score."
    ),
    "decoder": (
        "A Transformer Encoder-Decoder for sentiment classification built on "
        "pre-trained GloVe embeddings. The encoder processes the input sequence, "
        "and the decoder attends to the encoder outputs to produce a sentiment "
        "prediction."
    ),
}


def create_model_card(
    model_type: str,
    tuning_result: dict[str, Any] | None = None,
) -> str:
    """Generate a HuggingFace model card (README.md) for a Sentimentizer model.

    The model card includes YAML frontmatter, a description of the model
    architecture, usage instructions, and optionally the tuning metrics
    from a ``TuningRunResult``.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        tuning_result: Optional dict with tuning metrics (as produced by
            ``TuningRunResult`` or loaded from the saved JSON). If provided,
            the model card includes a metrics table.

    Returns:
        The model card content as a string.

    Raises:
        ValueError: If *model_type* is not recognized.
    """
    if model_type not in _MODEL_DESCRIPTIONS:
        valid_types = ", ".join(_MODEL_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown model type: {model_type!r}. Must be one of: {valid_types}")

    description = _MODEL_DESCRIPTIONS[model_type]
    repo_id = HF_WEIGHTS_REPOS.get(model_type, f"ryeyoo/sentimentizer-{model_type}")

    # ── YAML frontmatter ──
    yaml_block = (
        "---\n"
        "language: en\n"
        "license: mit\n"
        "tags:\n"
        "  - sentiment-analysis\n"
        "  - text-classification\n"
        f"  - {model_type}\n"
        "library_name: sentimentizer\n"
        "task: text-classification\n"
        "---\n"
    )

    # ── Title and description ──
    title = f"# Sentimentizer {model_type.upper()} Sentiment Model"
    desc_section = f"\n## Description\n\n{description}\n"

    # ── Training data ──
    training_section = (
        "\n## Training Data\n\n"
        "Trained on the [Yelp Open Dataset](https://www.yelp.com/dataset) reviews, "
        "with GloVe Wiki-Gigaword-100 pre-trained embeddings. Reviews are tokenized "
        "with a custom dictionary (20k vocab, min frequency 3) and padded/truncated "
        "to 200 tokens.\n"
    )

    # ── Metrics section (from tuning result) ──
    metrics_section = ""
    if tuning_result is not None:
        metrics_section = _format_metrics_section(tuning_result)

    # ── Usage section ──
    usage_section = _format_usage_section(model_type, repo_id)

    # ── Files section ──
    files_section = (
        "\n## Files\n\n"
        f"- `{model_type}_weights.pth` — Model state dictionary\n"
        "- `yelp.dictionary` — Gensim dictionary for tokenization\n"
    )

    # ── Combine ──
    return (
        yaml_block
        + title
        + desc_section
        + training_section
        + metrics_section
        + usage_section
        + files_section
    )


def _format_metrics_section(tuning_result: dict[str, Any]) -> str:
    """Format tuning metrics into a Markdown section for the model card."""
    lines = ["\n## Metrics\n"]

    # Table of key metrics
    table_lines = [
        "| Metric | Value |",
        "|--------|------|",
    ]

    metric_keys = [
        ("best_accuracy", "Accuracy"),
        ("best_loss", "Loss"),
        ("best_precision", "Precision"),
        ("best_recall", "Recall"),
        ("best_f1", "F1"),
        ("best_cohen_kappa", "Cohen's Kappa"),
        ("best_positive_accuracy", "Positive Accuracy"),
        ("best_negative_accuracy", "Negative Accuracy"),
    ]

    for key, label in metric_keys:
        val = tuning_result.get(key)
        if val is not None:
            if isinstance(val, float):
                table_lines.append(f"| {label} | {val:.4f} |")
            else:
                table_lines.append(f"| {label} | {val} |")

    lines.append("\n" + "\n".join(table_lines) + "\n")

    # Validation status
    validation_passed = tuning_result.get("validation_passed")
    if validation_passed is not None:
        status = "✅ Passed" if validation_passed else "❌ Failed"
        lines.append(f"\n**Validation:** {status}")

    # Tuning mode and iterations
    mode = tuning_result.get("mode", "")
    iterations = tuning_result.get("iterations_completed")
    converged = tuning_result.get("converged")
    elapsed = tuning_result.get("elapsed_seconds")

    meta_parts: list[str] = []
    if mode:
        meta_parts.append(f"Mode: `{mode}`")
    if iterations is not None:
        meta_parts.append(f"Iterations: {iterations}")
    if converged is not None:
        meta_parts.append(f"Converged: {'Yes' if converged else 'No'}")
    if elapsed is not None:
        meta_parts.append(f"Elapsed: {elapsed:.1f}s")

    if meta_parts:
        lines.append("\n" + " | ".join(meta_parts) + "\n")

    # Best config
    best_config = tuning_result.get("best_config")
    if best_config and isinstance(best_config, dict):
        config_lines = ["\n<details><summary>Best Configuration</summary>\n", "```json"]
        import json

        config_lines.append(json.dumps(best_config, indent=2))
        config_lines.append("```\n</details>\n")
        lines.extend(config_lines)

    return "".join(lines)


def _upload_model_card(
    api: HfApi,
    repo_id: str,
    model_type: str,
    tuning_result: dict[str, Any] | None,
) -> None:
    """Generate and upload a model card (README.md) to the Hugging Face Hub.

    Writes the model card content to a temporary file and uploads it
    as ``README.md`` to the repository.

    Args:
        api: HfApi instance for uploading.
        repo_id: Hugging Face repository ID.
        model_type: Model type for the model card.
        tuning_result: Optional tuning metrics to include.
    """
    try:
        model_card_content = create_model_card(model_type, tuning_result=tuning_result)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(model_card_content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update {model_type} model card",
        )
        logger.info(f"Successfully pushed model card to {repo_id}/README.md")

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to push model card: {e}")
        # Clean up temp file on error too
        try:
            if "tmp_path" in dir():
                Path(tmp_path).unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass


def _format_usage_section(model_type: str, repo_id: str) -> str:
    """Format usage instructions for the model card."""
    return (
        "\n## Usage\n\n"
        "```python\n"
        "from sentimentizer.hf import download_weights\n"
        "from sentimentizer.config import DriverConfig, weights_path_for\n\n"
        f"# Download weights + dictionary from Hugging Face Hub\n"
        f'weights_path = weights_path_for("{model_type}")\n'
        f"download_weights(\n"
        f'    "{model_type}",\n'
        f"    weights_path,\n"
        f"    dict_path=DriverConfig.files.dictionary_file_path,\n"
        f")\n\n"
        f"# Load and run inference\n"
        f"from sentimentizer.models.{model_type} import get_trained_model\n"
        f"from sentimentizer.tokenizer import get_trained_tokenizer\n\n"
        f'model = get_trained_model(device="cpu")\n'
        f"tokenizer = get_trained_tokenizer()\n\n"
        f"import numpy as np\n"
        f'token_ids = tokenizer.tokenize_text("amazing food great service")\n'
        f"score = model.predict(token_ids)\n"
        f"print(f'Sentiment score: {{score.item():.4f}}')  # >0.5 = positive, <0.5 = negative\n"
        "```\n"
    )


def push_model_to_hub(
    local_path: str | Path,
    model_type: str,
    repo_id: str | None = None,
    dict_path: str | Path | None = None,
    tuning_result: dict[str, Any] | None = None,
) -> None:
    """Upload model weights, dictionary, and model card to the Hugging Face Hub.

    If *repo_id* is provided, it uploads to that repository.
    Otherwise, it looks up the per-model repository from ``HF_WEIGHTS_REPOS``.

    When *tuning_result* is provided, a model card (README.md) is generated
    with the tuning metrics and uploaded alongside the weights.

    Args:
        local_path: Path to the local .pth weight file.
        model_type: Model type ('rnn', 'encoder', or 'decoder') used for the filename.
        repo_id: Optional Hugging Face repository ID.
        dict_path: Optional path to the dictionary file to also upload.
        tuning_result: Optional dict with tuning metrics to include in the model card.
    """
    if repo_id is None:
        repo_id = HF_WEIGHTS_REPOS.get(model_type)

    if repo_id is None:
        logger.error(f"No Hugging Face repo configured for model type {model_type!r}")
        return

    path = Path(local_path)
    if not path.exists():
        logger.error(f"Local weight file not found: {path}")
        return

    filename = f"{model_type}_weights.pth"
    api = HfApi()

    logger.info(f"Pushing {filename} to Hugging Face Hub: {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=f"Update {model_type} weights",
        )
        logger.info(f"Successfully pushed weights to {repo_id}/{filename}")

        if dict_path is not None:
            dict_file = Path(dict_path)
            if dict_file.exists():
                logger.info(f"Pushing dictionary to Hugging Face Hub: {repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(dict_file),
                    path_in_repo=dict_file.name,
                    repo_id=repo_id,
                    commit_message=f"Update dictionary for {model_type}",
                )
                logger.info(f"Successfully pushed dictionary to {repo_id}/{dict_file.name}")
            else:
                logger.error(f"Dictionary file not found: {dict_file}")

        # Upload model card (README.md) with tuning metrics
        _upload_model_card(api, repo_id, model_type, tuning_result)

    except Exception as e:
        # Check if error is because repo doesn't exist
        if "Repository Not Found" in str(e) or "404 Client Error" in str(e):
            logger.info(f"Repository {repo_id} not found. Attempting to create it...")
            try:
                api.create_repo(repo_id=repo_id, exist_ok=True)
                # Retry upload after creation
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    commit_message=f"Initial {model_type} weights upload",
                )
                logger.info(f"Successfully created repo and pushed weights to {repo_id}/{filename}")

                if dict_path is not None:
                    dict_file = Path(dict_path)
                    if dict_file.exists():
                        api.upload_file(
                            path_or_fileobj=str(dict_file),
                            path_in_repo=dict_file.name,
                            repo_id=repo_id,
                            commit_message=f"Initial dictionary upload for {model_type}",
                        )
                        logger.info(f"Successfully pushed dictionary to {repo_id}/{dict_file.name}")

                # Upload model card after creating new repo
                _upload_model_card(api, repo_id, model_type, tuning_result)

                return
            except Exception as create_error:
                logger.error(f"Failed to create repository or retry upload: {create_error}")
                return

        logger.error(f"Failed to push weights to Hugging Face Hub: {e}")


def pull_model_from_hub(
    repo_id: str,
    model_type: str,
    local_path: str | Path,
    dict_path: str | Path | None = None,
) -> bool:
    """Download model weights and optionally the dictionary from the Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/repo').
        model_type: Model type ('rnn', 'encoder', or 'decoder') used for the filename.
        local_path: Local path where the weights should be saved.
        dict_path: Optional local path where the dictionary should be saved.

    Returns:
        True if the weight download was successful, False otherwise.
    """
    filename = f"{model_type}_weights.pth"
    local_path = Path(local_path)

    logger.info(f"Pulling {filename} from Hugging Face Hub: {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        # Ensure target directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the downloaded file to the expected local path
        import shutil

        shutil.copy(downloaded_path, local_path)

        logger.info(f"Successfully pulled weights to {local_path}")

        if dict_path is not None:
            dict_file = Path(dict_path)
            try:
                logger.info(f"Pulling dictionary from Hugging Face Hub: {repo_id}...")
                downloaded_dict = hf_hub_download(
                    repo_id=repo_id,
                    filename=dict_file.name,
                )
                dict_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(downloaded_dict, dict_file)
                logger.info(f"Successfully pulled dictionary to {dict_file}")
            except EntryNotFoundError:
                logger.warning(
                    f"Dictionary file {dict_file.name} not found in repository {repo_id}"
                )
            except Exception as e:
                logger.error(f"Failed to pull dictionary from Hugging Face Hub: {e}")

        return True
    except EntryNotFoundError:
        logger.warning(f"Weights file {filename} not found in repository {repo_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to pull weights from Hugging Face Hub: {e}")
        return False


def download_weights(
    model_type: str,
    local_path: str | Path,
    repo_id: str | None = None,
    dict_path: str | Path | None = None,
) -> Path | None:
    """Download model weights and optionally the dictionary from the Hugging Face Hub.

    If *repo_id* is provided, it downloads from that repository.
    Otherwise, it looks up the per-model repository from ``HF_WEIGHTS_REPOS``.

    Args:
        model_type: One of 'rnn', 'encoder', or 'decoder'.
        local_path: Destination path on the local filesystem.
        repo_id: Optional Hugging Face repository ID.
        dict_path: Optional destination path for the dictionary on the local filesystem.

    Returns:
        The path to the downloaded weights file, or ``None`` if the download
        failed (e.g. the repo or file doesn't exist, or there is no network).
    """
    if repo_id is None:
        repo_id = HF_WEIGHTS_REPOS.get(model_type)

    if repo_id is None:
        logger.error(f"No Hugging Face repo configured for model type {model_type!r}")
        return None

    local_path = Path(local_path)
    filename = f"{model_type}_weights.pth"

    logger.info(f"Downloading {filename} from {repo_id} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        # Ensure target directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the downloaded file to the expected local path
        import shutil

        shutil.copy2(downloaded_path, local_path)
        logger.info(f"Weights downloaded to {local_path}")

        if dict_path is not None:
            dict_file = Path(dict_path)
            try:
                logger.info(f"Downloading dictionary from {repo_id} ...")
                downloaded_dict = hf_hub_download(
                    repo_id=repo_id,
                    filename=dict_file.name,
                )
                dict_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_dict, dict_file)
                logger.info(f"Dictionary downloaded to {dict_file}")
            except EntryNotFoundError:
                logger.warning(f"Dictionary file {dict_file.name} not found in {repo_id}")
            except Exception as e:
                logger.error(f"Failed to download dictionary from Hugging Face Hub: {e}")

        return local_path

    except EntryNotFoundError:
        logger.warning(f"Weights file {filename} not found in {repo_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to download weights from Hugging Face Hub: {e}")
        return None
