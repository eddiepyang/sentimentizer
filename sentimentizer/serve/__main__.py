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
    args = parser.parse_args()

    if args.config:
        os.environ["SENTIMENTIZER_SERVE_CONFIG"] = args.config

    from sentimentizer.serve.app import main

    main(host=args.host, port=args.port)
