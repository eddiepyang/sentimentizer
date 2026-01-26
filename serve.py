import torch
import numpy as np
from ray import serve
from starlette.requests import Request
from typing import Dict

from gensim import corpora
from sentimentizer.models.rnn import get_trained_model
from sentimentizer import config, tokenizer


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class SentimentDeployment:
    """deployment server for models"""

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_trained_model(batch_size=1, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer.get_trained_tokenizer()

    async def __call__(self, http_request: Request) -> Dict:
        # Get JSON data from the request
        json_input = await http_request.json()
        text = json_input.get("text", "")

        if not text:
            return {"error": "No text provided"}

        # Preprocess text to numpy
        processed_input = self.tokenizer.tokenize_text(text)

        prediction_tensor = self.model.predict(processed_input)
        score = prediction_tensor.item()

        return {
            "text": text,
            "sentiment_score": score,
            "prediction": "positive" if score > 0.5 else "negative",
        }


# Create the entry point for Ray Serve
app = SentimentDeployment.bind()
