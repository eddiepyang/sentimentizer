"""Shared helper utilities: model config, model loading, and path/parquet utils.

Heavy imports (torch, config) are DEFERRED to function bodies to avoid
importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from workflows.lifecycle import State

# ── Path utilities ───────────────────────────────────────────────


def _remove_path(path: str) -> None:
    """Remove a file or directory at the given path.

    Ray Data writes parquet as a directory of part files, but a previous
    run may have left a regular file at the same location.  This helper
    handles both cases so write_parquet never hits FileExistsError.
    """
    p = Path(path)
    if p.is_file() or p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _parquet_row_count(path: str) -> int:
    """Return the number of rows in a parquet file or directory from metadata.

    Returns 0 if the path does not exist.
    """
    import pyarrow.parquet as pq

    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return pq.read_metadata(str(p)).num_rows
    if p.is_dir():
        total = 0
        for f in sorted(p.glob("*.parquet")):
            total += pq.read_metadata(str(f)).num_rows
        return total
    return 0


# ── Model config ────────────────────────────────────────────────


def _get_model_config(model_type: str) -> Any:
    """Get the model config for the given model type."""
    from sentimentizer.config import DriverConfig

    if model_type == "rnn":
        return DriverConfig.rnn()
    elif model_type == "encoder":
        return DriverConfig.encoder()
    elif model_type == "decoder":
        return DriverConfig.decoder()
    else:
        raise ValueError(f"no matching model config for {model_type}")


def _load_model(state: State, device: str) -> Any:
    """Load a model, either fresh (new) or from checkpoint (update)."""

    from sentimentizer.config import DriverConfig

    model_config = _get_model_config(state.model)

    if state.model == "rnn":
        from sentimentizer.models.rnn import get_trained_model, new_model
    elif state.model == "encoder":
        from sentimentizer.models.encoder import get_trained_model, new_model
    elif state.model == "decoder":
        from sentimentizer.models.decoder import get_trained_model, new_model
    else:
        raise ValueError(f"no matching model for {state.model}")

    if not Path(DriverConfig.files.dictionary_file_path).exists():
        raise FileNotFoundError(
            f"Dictionary file not found at {DriverConfig.files.dictionary_file_path}. "
            "Ensure the tokenization step completed successfully before loading the model."
        )

    if state.run_type == "new":
        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            input_len=DriverConfig.tokenizer.max_len,
            model_config=model_config,
        )
    elif state.run_type == "update":
        model = get_trained_model(
            device=device,
            model_config=model_config,
        )
    else:
        raise ValueError(f"invalid run_type: {state.run_type}")

    return model
