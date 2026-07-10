import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Sentimentizer Serve")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to serve_config.yaml (default: bundled config, "
        "or SENTIMENTIZER_SERVE_CONFIG env var). "
        "Must be set before module imports, so it's promoted to "
        "SENTIMENTIZER_SERVE_CONFIG.",
    )
    env_diffusion = os.environ.get("SENTIMENTIZER_DIFFUSION_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    env_embeddings = os.environ.get("SENTIMENTIZER_EMBEDDINGS_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        default=env_diffusion,
        help="Enable image generation (SD3.5/FLUX.2 Klein/SDXL) endpoints. "
        "Requires GPU hardware and model weights. "
        "Can also be enabled via config (flux2_klein_enabled, sd35_enabled, "
        "sdxl_models) or via SENTIMENTIZER_DIFFUSION_ENABLED env var.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        default=env_embeddings,
        help="Enable dense embedding routes. BGE-M3 additionally requires "
        "SENTIMENTIZER_BGE_M3_ENABLED=1.",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["SENTIMENTIZER_SERVE_CONFIG"] = args.config
    if args.embeddings:
        os.environ["SENTIMENTIZER_EMBEDDINGS_ENABLED"] = "1"

    from sentimentizer.serve.app import main

    main(host=args.host, port=args.port, diffusion=args.diffusion)
