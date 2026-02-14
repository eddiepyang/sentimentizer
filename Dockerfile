FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY sentimentizer/ ./sentimentizer/

# Install the project + CPU-only PyTorch + Ray Serve
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and app from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Ray Serve default port
EXPOSE 8000
# Ray dashboard (optional)
EXPOSE 8265

ENV RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1

CMD ["serve", "run", "sentimentizer.serve:app", "--host", "0.0.0.0", "--port", "8000"]