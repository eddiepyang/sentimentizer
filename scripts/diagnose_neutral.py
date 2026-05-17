#!/usr/bin/env python3
"""Diagnose how models score 3-star (neutral) reviews.

Loads the trained RNN, Encoder, and Decoder models and runs inference on
samples of 1-star, 2-star, 3-star, 4-star, and 5-star reviews.
Prints per-star per-class probability distributions and a simple ASCII
histogram so we can see where neutral reviews cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sentimentizer.config import FileConfig, LABEL_NAMES
from sentimentizer.models.rnn import get_trained_model as get_rnn
from sentimentizer.models.encoder import get_trained_model as get_encoder
from sentimentizer.models.decoder import get_trained_model as get_decoder
from sentimentizer.tokenizer import get_trained_tokenizer


def _ascii_histogram(scores: np.ndarray, bins: int = 10, width: int = 50) -> str:
    """Simple ASCII histogram for quick terminal inspection."""
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    max_count = max(counts.max(), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / max_count)
        lines.append(f"  {edges[i]:.2f}-{edges[i + 1]:.2f} | {bar} {c}")
    return "\n".join(lines)


def main() -> None:
    device = "cpu"  # diagnostic script; CPU is fine
    print("Loading tokenizer …")
    tokenizer = get_trained_tokenizer()

    print("Loading models …")
    models = {
        "rnn": get_rnn(device),
        "encoder": get_encoder(device),
        "decoder": get_decoder(device),
    }

    print("Loading raw reviews …")
    df = pd.read_parquet(FileConfig.raw_reviews_file_path)

    # Ensure tokens are lists (parquet may store as numpy arrays)
    df["tokens"] = df["tokens"].apply(lambda x: list(x) if not isinstance(x, list) else x)

    # Sample sizes per star rating
    n_per_star = 2000

    samples: dict[int, pd.DataFrame] = {}
    for star in [1, 2, 3, 4, 5]:
        star_df = df[df["stars"] == float(star)]
        if len(star_df) > n_per_star:
            star_df = star_df.sample(n=n_per_star, random_state=42)
        samples[star] = star_df.reset_index(drop=True)
        print(f"  {star}-star: {len(star_df)} reviews")

    # Pre-tokenize all texts to numeric arrays (same pipeline as training)
    print("Pre-tokenizing texts …")
    all_texts: list[str] = []
    all_stars: list[int] = []
    for star in [1, 2, 3, 4, 5]:
        texts = samples[star]["text"].tolist()
        all_texts.extend(texts)
        all_stars.extend([star] * len(texts))

    # Convert texts to token-id arrays via tokenizer.tokenize_text
    token_arrays: list[np.ndarray] = []
    for text in all_texts:
        arr = tokenizer.tokenize_text(text)  # shape (1, max_len)
        token_arrays.append(arr)

    # Stack into a single tensor for batched inference
    batch_size = 256
    # Per-class scores per model: {model_name: {label: [scores]}}
    all_probs: dict[str, dict[str, list[float]]] = {
        name: {label: [] for label in LABEL_NAMES} for name in models
    }

    print("Running inference …")
    with torch.no_grad():
        for model_name, model in models.items():
            model.eval()
            for i in range(0, len(token_arrays), batch_size):
                batch = np.vstack(token_arrays[i : i + batch_size])  # (B, max_len)
                batch_t = torch.from_numpy(batch).to(device)
                logits = model(batch_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, num_classes)
                for j in range(probs.shape[0]):
                    for k, label in enumerate(LABEL_NAMES):
                        all_probs[model_name][label].append(float(probs[j, k]))

    # Build a results DataFrame
    results_data: dict[str, list] = {"stars": all_stars, "text": all_texts}
    for model_name in models:
        for label in LABEL_NAMES:
            results_data[f"{model_name}_{label}"] = all_probs[model_name][label]
    results = pd.DataFrame(results_data)

    print("\n" + "=" * 70)
    print("PER-CLASS SCORE DISTRIBUTIONS BY STAR RATING")
    print("=" * 70)

    for star in [1, 2, 3, 4, 5]:
        subset = results[results["stars"] == star]
        print(f"\n--- {star}-star reviews (n={len(subset)}) ---")
        for model_name in models:
            print(f"  {model_name}:")
            for label in LABEL_NAMES:
                scores = subset[f"{model_name}_{label}"].values
                print(
                    f"    {label:8s}: mean={scores.mean():.4f}  std={scores.std():.4f}  "
                    f"p25={np.percentile(scores, 25):.4f}  "
                    f"median={np.median(scores):.4f}  "
                    f"p75={np.percentile(scores, 75):.4f}"
                )

    print("\n" + "=" * 70)
    print("3-STAR (NEUTRAL) PROBABILITY HISTOGRAMS")
    print("=" * 70)
    neutral = results[results["stars"] == 3]
    for model_name in models:
        print(f"\n{model_name.upper()} — 3-star neutral probability:")
        print(_ascii_histogram(neutral[f"{model_name}_neutral"].values, bins=10, width=50))

    # Overlap analysis: how do neutral review probabilities compare with
    # negative and positive review probabilities?
    print("\n" + "=" * 70)
    print("NEUTRAL CLASS PROBABILITY OVERLAP ANALYSIS")
    print("=" * 70)
    for model_name in models:
        neg = results[results["stars"].isin([1, 2])][f"{model_name}_neutral"].values
        pos = results[results["stars"].isin([4, 5])][f"{model_name}_neutral"].values
        neu = neutral[f"{model_name}_neutral"].values

        neg_q1, neg_q3 = np.percentile(neg, [25, 75])
        pos_q1, pos_q3 = np.percentile(pos, [25, 75])

        in_neg_iqr = ((neu >= neg_q1) & (neu <= neg_q3)).mean()
        in_pos_iqr = ((neu >= pos_q1) & (neu <= pos_q3)).mean()
        in_gap = ((neu > neg_q3) & (neu < pos_q1)).mean()

        print(f"\n{model_name.upper()} neutral-class probability:")
        print(f"  3-star inside negative IQR  [{neg_q1:.4f}, {neg_q3:.4f}]: {in_neg_iqr:.1%}")
        print(f"  3-star inside positive IQR  [{pos_q1:.4f}, {pos_q3:.4f}]: {in_pos_iqr:.1%}")
        print(f"  3-star in gap between them: {in_gap:.1%}")

    # Neutral detection rate: fraction of 3-star reviews where neutral is
    # the highest-probability class
    print("\n" + "=" * 70)
    print("NEUTRAL DETECTION RATE")
    print("=" * 70)
    for model_name in models:
        prob_cols = [f"{model_name}_{label}" for label in LABEL_NAMES]
        predicted = neutral[prob_cols].values.argmax(axis=1)  # 0=neg, 1=neu, 2=pos
        neutral_rate = (predicted == 1).mean()
        positive_rate = (predicted == 2).mean()
        negative_rate = (predicted == 0).mean()
        print(
            f"  {model_name}: neutral={neutral_rate:.1%}  "
            f"positive={positive_rate:.1%}  negative={negative_rate:.1%}"
        )

    # Save raw results for further analysis
    out_path = "/tmp/neutral_diagnostic_scores.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nRaw scores saved to: {out_path}")


if __name__ == "__main__":
    main()
