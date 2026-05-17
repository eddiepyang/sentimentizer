import argparse
from sentimentizer.serve.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Sentimentizer Serve")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
