FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY sentimentizer/ ./sentimentizer/

# Install the project + CPU-only PyTorch + Ray Serve
# Both installs use the CPU-only PyTorch index to avoid pulling CUDA wheels
RUN uv pip install --system --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --default-index https://pypi.org/simple/ .

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (K8s runAsNonRoot compliance)
RUN useradd -r -s /bin/false -d /app sentimentizer \
    && chown -R sentimentizer:sentimentizer /app

# Copy installed packages and app from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

USER sentimentizer

# Ray Serve default port
EXPOSE 8000
# Ray dashboard (optional)
EXPOSE 8265

CMD ["serve", "run", "sentimentizer.serve:app", "--host", "0.0.0.0", "--port", "8000"]