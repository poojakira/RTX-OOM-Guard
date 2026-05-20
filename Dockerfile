# High-Performance ML Infrastructure: rtx-oom-guard
# Optimized for NVIDIA-based GPU Workloads

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="poojakira"
LABEL description="rtx-oom-guard: Predictive GPU Memory Defragmenter"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (Triton/CUDA development tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace
COPY . /app

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e "."

# Create non-root user and set ownership
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose Telemetry / Dashboard Ports
EXPOSE 8000
EXPOSE 5173

# Entry point starts the infrastructure monitor with the default config
CMD ["python", "run.py", "--config", "configs/config.yaml"]
