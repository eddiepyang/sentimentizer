"""Tokenize stage: build/update dictionary, write processed parquet.

Heavy imports (torch, ray, gensim, config) are DEFERRED to function bodies to
avoid importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

from workflows.helpers import _parquet_row_count, _remove_path
from workflows.lifecycle import State, _ensure_ray_initialized, logger


def run_tokenize(state: State, *, resume: bool = False) -> None:
    """Build/update dictionary and write processed parquet."""
    from sentimentizer.config import DriverConfig
    from sentimentizer.tokenizer import Tokenizer, regex_tokenize
    from workflows.lifecycle import is_ray_available

    _ensure_ray_initialized()

    # For 'new' runs, always (re)create the dictionary and re-tokenize
    if state.run_type == "new":
        if is_ray_available():
            import ray

            reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
            tokenizer = Tokenizer.from_dataset(reviews_data)

            processed_ds = tokenizer.transform_dataset(reviews_data)
            _remove_path(DriverConfig.files.processed_reviews_file_path)
            processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
        else:
            import os

            import pandas as pd

            reviews_data = pd.read_parquet(DriverConfig.files.raw_reviews_file_path)
            tokenizer = Tokenizer.from_data(reviews_data)

            processed_df = tokenizer.transform_dataframe(reviews_data)
            _remove_path(DriverConfig.files.processed_reviews_file_path)
            os.makedirs(
                os.path.dirname(DriverConfig.files.processed_reviews_file_path), exist_ok=True
            )
            processed_df.to_parquet(
                DriverConfig.files.processed_reviews_file_path, engine="pyarrow"
            )

    elif resume or state.run_type == "update":
        # Skip tokenization if the processed parquet already has enough rows
        from sentimentizer.config import TokenizerConfig

        stop = TokenizerConfig.stop
        existing_rows = _parquet_row_count(DriverConfig.files.processed_reviews_file_path)
        if existing_rows >= stop:
            logger.info(f"skipping tokenize: {existing_rows} rows already exist (need {stop})")
            return

        if is_ray_available():
            import ray
            from gensim import corpora

            reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
            dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
            tokenizer = Tokenizer(dictionary=dictionary)
            if resume:
                logger.info(
                    f"resuming from checkpoint: updating dictionary from "
                    f"{DriverConfig.files.dictionary_file_path}"
                )
                tokenizer.update_from_dataset(reviews_data)

            processed_ds = tokenizer.transform_dataset(reviews_data)
            _remove_path(DriverConfig.files.processed_reviews_file_path)
            processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
        else:
            import os

            import pandas as pd
            from gensim import corpora

            reviews_data = pd.read_parquet(DriverConfig.files.raw_reviews_file_path)
            dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
            tokenizer = Tokenizer(dictionary=dictionary)
            if resume:
                logger.info(
                    f"resuming from checkpoint: updating dictionary from "
                    f"{DriverConfig.files.dictionary_file_path}"
                )
                from sentimentizer.config import TokenizerConfig

                t_cfg = TokenizerConfig()
                texts = reviews_data[t_cfg.text_col].apply(
                    lambda x: (
                        x
                        if isinstance(x, list)
                        else list(x) if hasattr(x, "__iter__") else regex_tokenize(str(x))
                    )
                )
                dictionary.add_documents(texts)
                if t_cfg.save_dictionary:
                    dictionary.save(DriverConfig.files.dictionary_file_path)
                    logger.info(
                        f"updated dictionary saved to {DriverConfig.files.dictionary_file_path}..."
                    )

            processed_df = tokenizer.transform_dataframe(reviews_data)
            _remove_path(DriverConfig.files.processed_reviews_file_path)
            os.makedirs(
                os.path.dirname(DriverConfig.files.processed_reviews_file_path), exist_ok=True
            )
            processed_df.to_parquet(
                DriverConfig.files.processed_reviews_file_path, engine="pyarrow"
            )
    else:
        raise ValueError(f"invalid run_type: {state.run_type}")
