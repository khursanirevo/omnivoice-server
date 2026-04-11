# --- Builder stage: install Python dependencies with CUDA toolkit ---
FROM pytorch/pytorch:2.3.0-cuda12.4-cudnn9-devel AS builder

WORKDIR /build

# Copy project files
COPY pyproject.toml README.md ./
COPY omnivoice_server ./omnivoice_server

# Install the package (PyTorch CUDA is already in the base image)
RUN pip install --no-cache-dir .

# --- Runtime stage: slim image with CUDA runtime libs + MPS control binary ---
FROM pytorch/pytorch:2.3.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install runtime dependencies (libsndfile1 for audio, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin/omnivoice-server /opt/conda/bin/omnivoice-server

# Copy the app source (needed for python -m execution)
COPY --from=builder /build/omnivoice_server ./omnivoice_server

# Copy nvidia-cuda-mps-control from the devel stage (not present in runtime images)
COPY --from=builder /usr/local/cuda/bin/nvidia-cuda-mps-control /usr/local/cuda/bin/nvidia-cuda-mps-control

# Create profile directory
RUN mkdir -p /app/profiles

# Expose server port
EXPOSE 8880

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

# Run server
CMD ["python", "-m", "omnivoice_server"]
