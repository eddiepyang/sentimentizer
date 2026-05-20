"""Router module for Yelp review categorization.

Provides a 3-class routing classifier (Dietary, Service, General)
trained via contrastive learning with sentence-transformers + sklearn.
"""

from sentimentizer.router.config import AugmentConfig, RouteLabels, RouterConfig, SetFitConfig
from sentimentizer.router.model import RouterModel
from sentimentizer.router.seeds import SEED_UTTERANCES

__all__ = [
    "AugmentConfig",
    "RouterConfig",
    "RouterModel",
    "SetFitConfig",
    "RouteLabels",
    "SEED_UTTERANCES",
]
