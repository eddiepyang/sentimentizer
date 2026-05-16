"""Sentiment analysis model architectures.

- BaseSentimentModel: Shared base class with predict() method
- RNN: Bidirectional LSTM
- Encoder: Transformer encoder with CLS token
- Decoder: Encoder-decoder transformer
"""

from sentimentizer.models.base import BaseSentimentModel

__all__ = ["BaseSentimentModel"]
