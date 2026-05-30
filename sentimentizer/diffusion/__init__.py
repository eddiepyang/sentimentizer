from sentimentizer.diffusion.config import DiffusionModelConfig
from sentimentizer.diffusion.image_utils import (
    _REF_MAX_PIXELS,
    b64_encode,
    decode_b64_image,
    encode_pil,
    generate_id,
)
from sentimentizer.diffusion.job_store import JobStore
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE
from sentimentizer.diffusion.predictor import (
    DiffusionPredictor,
    DiffusionPredictorProtocol,
    Flux2KleinPredictor,
    SD35Predictor,
    SDXLPredictor,
    create_predictor,
)

__all__ = [
    "DiffusionModelConfig",
    "DiffusionPredictor",
    "DiffusionPredictorProtocol",
    "Flux2KleinPredictor",
    "JobStore",
    "SD35Predictor",
    "SDXLPredictor",
    "create_predictor",
    "_REF_MAX_PIXELS",
    "b64_encode",
    "decode_b64_image",
    "encode_pil",
    "generate_id",
]

if MFLUX_AVAILABLE:
    from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor  # noqa: F401

    __all__.append("MLXFlux2KleinPredictor")
