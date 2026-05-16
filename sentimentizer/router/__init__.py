"""SetFit router module for Yelp review categorization.

Provides a 3-class routing classifier (Dietary, Service, General)
trained via contrastive learning with SetFit.
"""

# Apply transformers compatibility shim before setfit imports.
# This ensures default_logdir is available when setfit 1.1.x
# tries to import it from transformers.training_args.
import sentimentizer.compat  # noqa: F401
from sentimentizer.router.config import AugmentConfig, RouteLabels, SetFitConfig
from sentimentizer.router.seeds import SEED_UTTERANCES

__all__ = ["AugmentConfig", "SetFitConfig", "RouteLabels", "SEED_UTTERANCES"]
