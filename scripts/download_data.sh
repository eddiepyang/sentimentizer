#!/usr/bin/env bash
set -e

# Determine the absolute paths based on config.py expectations
# The config sets external_data = root.parent / "data"
# where root is the sentimentizer project directory.
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="$PROJECT_ROOT/../data"

echo "Creating data directory at: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo "=========================================================="
echo "GloVe Embeddings"
echo "=========================================================="
echo "GloVe embeddings are now auto-downloaded via gensim.downloader"
echo "on first training run. They are cached to ~/gensim-data/ and"
echo "only downloaded once."
echo ""
echo "To pre-download manually, run:"
echo "  uv run python -c \"import gensim.downloader as api; api.load('glove-wiki-gigaword-100')\""
echo ""

echo "=========================================================="
echo "Yelp Open Dataset Instructions"
echo "=========================================================="
echo "The official Yelp dataset (yelp_dataset.tar) cannot be downloaded"
echo "automatically via a simple URL because it requires accepting an agreement."
echo ""
echo "Please follow these steps:"
echo "1. Go to https://www.yelp.com/dataset"
echo "2. Accept the terms and download the dataset."
echo "3. Rename and move the downloaded file to: $DATA_DIR/yelp_dataset.tar"
echo "=========================================================="
echo ""
echo "Alternatively, if you have the Kaggle CLI installed and configured,"
echo "you can download a mirror of the dataset:"
echo "  kaggle datasets download yelp-dataset/yelp-dataset -p $DATA_DIR"
echo "  unzip $DATA_DIR/yelp-dataset.zip -d $DATA_DIR"
echo "  tar -cvf $DATA_DIR/yelp_dataset.tar -C $DATA_DIR yelp_academic_dataset_review.json"
echo "=========================================================="
echo "Setup complete!"
