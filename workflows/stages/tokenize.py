"""Tokenize stage: build/update dictionary, write processed parquet.

Heavy imports (torch, ray, gensim, config) are DEFERRED to function bodies to
avoid importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

from workflows.helpers import _parquet_row_count, _remove_path
from workflows.lifecycle import State, _ensure_ray_initialized, logger


def run_tokenize(state: State, *, resume: bool = False) -> None:
    """Build/update dictionary and write processed parquet.

    For GloVe models (rnn, encoder, decoder), builds the gensim dictionary
    and writes GloVe-tokenized data to ``processed_reviews_file_path``.
    For HF models (modernbert), skips GloVe tokenization and instead
    tokenizes with the Hugging Face tokenizer, writing to
    ``hf_processed_reviews_file_path``.
    """
    from sentimentizer.models.base import _MODEL_REGISTRY, _ensure_registry

    _ensure_registry()
    model_cls = _MODEL_REGISTRY[state.model]["model_class"]

    if not getattr(model_cls, "NEEDS_TOKENIZE_STAGE", True):
        # HF models don't need GloVe tokenization — they need HF tokenization
        _run_hf_tokenize(state)
        return

    from gensim import corpora

    from sentimentizer.config import DriverConfig, TokenizerConfig
    from sentimentizer.data_source import read_parquet
    from sentimentizer.tokenizer import Tokenizer
    from workflows.lifecycle import is_ray_available

    _ensure_ray_initialized()

    stop = TokenizerConfig.stop
    existing_rows = _parquet_row_count(DriverConfig.files.processed_reviews_file_path)
    skip_data = existing_rows >= stop

    # A "new" run should always re-transform data from scratch — the
    # dictionary, tokenizer config (e.g. include_neutral), or class
    # mapping may have changed since the last run, so stale parquet
    # files with old targets would break 3-class training.
    if state.run_type == "new":
        skip_data = False

    if skip_data and (resume or state.run_type == "update"):
        logger.info(f"skipping tokenize: {existing_rows} rows already exist (need {stop})")
        return

    use_ray = is_ray_available()

    # --- Read raw data as DataSource (unified I/O) -------------------------
    data_source = read_parquet(DriverConfig.files.raw_reviews_file_path, use_ray=use_ray)

    if state.run_type == "new":
        # Always build dictionary from scratch
        tokenizer = Tokenizer.build_dictionary(data_source)

    elif resume or state.run_type == "update":
        dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
        tokenizer = Tokenizer(dictionary=dictionary)
        if resume:
            logger.info(
                f"resuming from checkpoint: updating dictionary from "
                f"{DriverConfig.files.dictionary_file_path}"
            )
            tokenizer.update_dictionary(data_source)
    else:
        raise ValueError(f"invalid run_type: {state.run_type}")

    # --- Transform and write (unified for both paths) ----------------------
    if not skip_data:
        processed = tokenizer.transform(data_source)
        _remove_path(DriverConfig.files.processed_reviews_file_path)
        processed.write_parquet(DriverConfig.files.processed_reviews_file_path)


def _run_hf_tokenize(state: State) -> None:
    """Tokenize raw reviews with Hugging Face tokenizer and write HF parquet.

    Used by models that don't need GloVe tokenization (e.g., modernbert).
    Reads raw reviews, applies HF tokenization, and writes to
    ``hf_processed_reviews_file_path`` with ``input_ids``, ``attention_mask``,
    and ``target`` columns.
    """
    from sentimentizer.config import DriverConfig, TokenizerConfig
    from sentimentizer.data_source import read_parquet
    from sentimentizer.hf_tokenizer import HFTokenizer
    from sentimentizer.models.base import _MODEL_REGISTRY, _ensure_registry
    from workflows.lifecycle import is_ray_available

    _ensure_registry()
    model_cls = _MODEL_REGISTRY[state.model]["model_class"]
    hf_model_name = getattr(model_cls, "HF_MODEL_NAME", None)
    if hf_model_name is None:
        raise ValueError(f"Model {state.model!r} has no HF_MODEL_NAME — cannot HF-tokenize")

    hf_path = DriverConfig.files.hf_processed_reviews_file_path

    # Check if HF-tokenized data already exists
    existing_rows = _parquet_row_count(hf_path)
    stop = TokenizerConfig.stop
    skip_data = existing_rows >= stop

    # A "new" run always re-tokenizes from scratch
    if state.run_type == "new":
        skip_data = False

    if skip_data and state.run_type == "update":
        logger.info(f"skipping HF tokenize: {existing_rows} rows already exist (need {stop})")
        return

    use_ray = is_ray_available()
    data_source = read_parquet(DriverConfig.files.raw_reviews_file_path, use_ray=use_ray)

    hf_tokenizer = HFTokenizer.from_pretrained(hf_model_name)
    processed = hf_tokenizer.transform(data_source)

    _remove_path(hf_path)
    processed.write_parquet(hf_path)
    logger.info(f"HF tokenization complete: wrote {hf_path}")
