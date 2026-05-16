"""Tests for sentimentizer.data_source module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentimentizer.config import TokenizerConfig
from sentimentizer.data_source import PandasDataSource, read_parquet
from sentimentizer.tokenizer import Tokenizer


class TestPandasDataSource:
    """Tests for PandasDataSource (no Ray required)."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stars": [1, 2, 3, 4, 5],
                "tokens": [["a", "b"], ["c"], ["d", "e", "f"], ["g"], ["h", "i"]],
                "target": [0.0, 0.0, 0.5, 1.0, 1.0],
            }
        )

    @pytest.fixture
    def binary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stars": [1, 2, 4, 5],
                "target": [0.0, 0.0, 1.0, 1.0],
            }
        )

    # --- Factory -----------------------------------------------------------

    def test_read_parquet(self, tmp_path) -> None:
        path = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(path)
        ds = PandasDataSource.read_parquet(str(path))
        pd.testing.assert_frame_equal(ds.to_pandas(), df)

    # --- Expression-based filter (Risk 9 mitigation) -------------------------

    def test_filter_expression_ne(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        filtered = ds.filter(col="stars", op="ne", value=3)
        assert filtered.count() == 4
        assert 3 not in filtered.to_pandas()["stars"].values

    def test_filter_expression_eq(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        filtered = ds.filter(col="stars", op="eq", value=3)
        assert filtered.count() == 1
        assert filtered.to_pandas()["stars"].iloc[0] == 3

    def test_filter_expression_symbolic(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        filtered = ds.filter(col="stars", op="!=", value=3)
        assert filtered.count() == 4

    def test_filter_expression_invalid_op(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        with pytest.raises(ValueError, match="Unsupported operator"):
            ds.filter(col="stars", op="bad", value=3)

    # --- Row-dict predicate filter -----------------------------------------

    def test_filter_predicate(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        filtered = ds.filter(lambda row: row["stars"] != 3)
        assert filtered.count() == 4

    def test_filter_both_styles_rejected(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        with pytest.raises(ValueError, match="Cannot specify both"):
            ds.filter(lambda r: True, col="stars", op="ne", value=3)

    def test_filter_no_args_rejected(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        with pytest.raises(ValueError, match="Must provide either"):
            ds.filter()

    # --- Map batches -------------------------------------------------------

    def test_map_batches_identity(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)

        def identity(batch: dict) -> dict:
            return batch

        result = ds.map_batches(identity, batch_size=2)
        assert result.count() == 5
        assert list(result.columns) == list(sample_df.columns)

    def test_map_batches_transform(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)

        def double_stars(batch: dict) -> dict:
            batch["stars"] = np.asarray(batch["stars"]) * 2
            return batch

        result = ds.map_batches(double_stars, batch_size=2)
        assert result.to_pandas()["stars"].tolist() == [2, 4, 6, 8, 10]

    def test_map_batches_schema_mismatch(self, sample_df) -> None:
        """Risk 17: schema drift should raise, not silently produce NaNs."""
        ds = PandasDataSource(sample_df)

        call_count = 0

        def bad_fn(batch: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                batch.pop("target")
            return batch

        with pytest.raises(ValueError, match="Schema mismatch"):
            ds.map_batches(bad_fn, batch_size=2)

    def test_map_batches_empty_result(self) -> None:
        ds = PandasDataSource(pd.DataFrame({"a": []}))
        result = ds.map_batches(lambda b: b, batch_size=10)
        assert result.count() == 0

    # --- Train / val split -------------------------------------------------

    def test_train_test_split(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        train, val = ds.train_test_split(test_size=0.4, random_state=42)
        assert train.count() == 3
        assert val.count() == 2

    # --- Balancing ---------------------------------------------------------

    def test_balance_positive_majority(self, binary_df) -> None:
        ds = PandasDataSource(binary_df)
        balanced = ds.balance(target_col="target", random_state=42)
        neg = (balanced.to_pandas()["target"] == 0.0).sum()
        pos = (balanced.to_pandas()["target"] == 1.0).sum()
        assert neg == pos == 2

    def test_balance_negative_majority(self) -> None:
        df = pd.DataFrame(
            {
                "stars": [1, 2, 3, 4],
                "target": [0.0, 0.0, 0.0, 1.0],
            }
        )
        ds = PandasDataSource(df)
        balanced = ds.balance(target_col="target", random_state=42)
        neg = (balanced.to_pandas()["target"] == 0.0).sum()
        pos = (balanced.to_pandas()["target"] == 1.0).sum()
        assert neg == pos == 1

    def test_balance_already_balanced(self, binary_df) -> None:
        ds = PandasDataSource(binary_df)
        balanced = ds.balance(target_col="target", random_state=42)
        assert balanced.count() == 4

    # --- Iteration ---------------------------------------------------------

    def test_iter_batches_numpy_format(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        batches = list(ds.iter_batches(batch_size=2, batch_format="numpy"))
        assert len(batches) == 3  # 5 rows / batch_size 2 = 3 batches
        assert isinstance(batches[0]["stars"], np.ndarray)

    def test_iter_batches_list_format(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        batches = list(ds.iter_batches(batch_size=2, batch_format="list"))
        assert len(batches) == 3
        assert isinstance(batches[0]["stars"], list)

    # --- Properties / conversion -------------------------------------------

    def test_count(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        assert ds.count() == 5

    def test_is_ray(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        assert not ds.is_ray

    def test_columns(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        assert set(ds.columns) == {"stars", "tokens", "target"}

    def test_to_pandas(self, sample_df) -> None:
        ds = PandasDataSource(sample_df)
        pd.testing.assert_frame_equal(ds.to_pandas(), sample_df)

    # --- I/O ---------------------------------------------------------------

    def test_write_parquet(self, sample_df, tmp_path) -> None:
        ds = PandasDataSource(sample_df)
        out_path = tmp_path / "out" / "test.parquet"
        ds.write_parquet(str(out_path))
        assert out_path.exists()
        read_back = pd.read_parquet(out_path)
        pd.testing.assert_frame_equal(read_back, sample_df)

    def test_read_parquet_factory(self, tmp_path) -> None:
        path = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2]})
        df.to_parquet(path)
        ds = read_parquet(str(path), use_ray=False)
        assert isinstance(ds, PandasDataSource)
        assert ds.count() == 2


class TestTokenizerUnified:
    """Tests for unified Tokenizer.build_dictionary and Tokenizer.transform."""

    @pytest.fixture
    def tokenized_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stars": [2, 5],
                "tokens": [
                    ["the", "chicken", "never", "showed", "up"],
                    ["the", "food", "was", "terrific"],
                ],
            }
        )

    def test_build_dictionary_from_data_source(self, tmp_path, tokenized_df) -> None:
        """Tokenizer.build_dictionary should work with PandasDataSource."""
        cfg = TokenizerConfig(
            text_col="tokens",
            label_col="stars",
            save_dictionary=False,
            dict_min=1,
            no_above=1.0,
        )
        ds = PandasDataSource(tokenized_df)
        tokenizer = Tokenizer.build_dictionary(ds, cfg=cfg)

        assert tokenizer.dictionary is not None
        assert len(tokenizer.dictionary) > 0
        # Common words should be in the dictionary
        assert "the" in tokenizer.dictionary.token2id

    def test_transform_data_source(self, tmp_path, tokenized_df) -> None:
        """Tokenizer.transform should produce numeric sequences from DataSource."""
        cfg = TokenizerConfig(
            text_col="tokens", label_col="stars", save_dictionary=False, dict_min=1
        )
        ds = PandasDataSource(tokenized_df)
        tokenizer = Tokenizer.build_dictionary(ds, cfg=cfg)

        result = tokenizer.transform(ds)
        result_df = result.to_pandas()

        # Neutral (3-star) reviews should be dropped — none here, so 2 rows
        assert result_df.shape[0] == 2
        assert "data" in result_df.columns
        assert "target" in result_df.columns

        # data column should contain numeric sequences
        assert isinstance(result_df["data"].iloc[0], (np.ndarray, list))
        assert len(result_df["data"].iloc[0]) == cfg.max_len

    def test_transform_drops_neutral(self) -> None:
        """Tokenizer.transform should drop 3-star reviews via expression filter."""
        df = pd.DataFrame(
            {
                "stars": [1, 3, 5],
                "tokens": [["a", "b"], ["c", "d"], ["e", "f"]],
            }
        )
        cfg = TokenizerConfig(
            text_col="tokens", label_col="stars", save_dictionary=False, dict_min=1
        )
        ds = PandasDataSource(df)
        tokenizer = Tokenizer.build_dictionary(ds, cfg=cfg)

        result = tokenizer.transform(ds)
        result_df = result.to_pandas()

        # 3-star review should be dropped
        assert result_df.shape[0] == 2
        # Only numeric columns remain after transform
        assert "stars" not in result_df.columns
        assert "data" in result_df.columns
        assert "target" in result_df.columns

    def test_update_dictionary(self, tokenized_df) -> None:
        """Tokenizer.update_dictionary should add new tokens without reassigning IDs."""
        cfg = TokenizerConfig(
            text_col="tokens", label_col="stars", save_dictionary=False, dict_min=1
        )
        ds = PandasDataSource(tokenized_df)
        tokenizer = Tokenizer.build_dictionary(ds, cfg=cfg)

        old_ids = dict(tokenizer.dictionary.token2id)
        old_len = len(tokenizer.dictionary)

        # New data with one unseen word
        new_df = pd.DataFrame(
            {
                "stars": [4],
                "tokens": [["zzzz", "the"]],
            }
        )
        new_ds = PandasDataSource(new_df)
        tokenizer.update_dictionary(new_ds)

        # Existing IDs should be preserved
        for word, idx in old_ids.items():
            assert tokenizer.dictionary.token2id[word] == idx

        # At least one new word should be added (may be 2 if "the" was filtered by no_above)
        assert len(tokenizer.dictionary) > old_len
        assert "zzzz" in tokenizer.dictionary.token2id
