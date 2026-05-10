"""Extract stage: raw reviews → parquet.

Heavy imports (torch, ray, config) are DEFERRED to function bodies to avoid
importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

from workflows.helpers import _parquet_row_count, _remove_path
from workflows.lifecycle import State, _ensure_ray_initialized, logger


def run_extract(state: State, *, stop: int) -> None:
    """Extract raw reviews into parquet."""
    _ensure_ray_initialized()

    from sentimentizer.config import DriverConfig
    from sentimentizer.extractor import extract_data_local, extract_data_ray
    from workflows.lifecycle import is_ray_available

    # Skip extraction if the raw parquet already has enough rows
    existing_rows = _parquet_row_count(DriverConfig.files.raw_reviews_file_path)
    if existing_rows >= stop:
        logger.info(f"skipping extract: {existing_rows} rows already exist (need {stop})")
        return

    if is_ray_available():
        ds = extract_data_ray(
            DriverConfig.files.archive_file_path,
            DriverConfig.files.raw_file_path,
            stop=stop,
        )
        _remove_path(DriverConfig.files.raw_reviews_file_path)
        ds.write_parquet(DriverConfig.files.raw_reviews_file_path)
    else:
        df = extract_data_local(
            DriverConfig.files.archive_file_path,
            DriverConfig.files.raw_file_path,
            stop=stop,
        )
        _remove_path(DriverConfig.files.raw_reviews_file_path)
        # Ensure parent directories exist
        import os

        os.makedirs(os.path.dirname(DriverConfig.files.raw_reviews_file_path), exist_ok=True)
        df.to_parquet(DriverConfig.files.raw_reviews_file_path, engine="pyarrow")
