FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgdal-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Install Pixi ────────────────────────────────────────────────────────────
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:$PATH"

# ── Set working directory and install deps ──────────────────────────────────
WORKDIR /workspace
COPY pixi.toml ./

RUN pixi install --verbose

# ── Install mmcv from OpenMMLab prebuilt wheels (not on PyPI) ────────────────
RUN /workspace/.pixi/envs/default/bin/pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html

EXPOSE 8888
CMD ["bash"]
