# Base Isaac Sim image
FROM nvcr.io/nvidia/isaac-sim:5.0.0

ENV DOCKER_USER_HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Utilities
RUN apt-get update && apt-get install -y \
    git python3 python3-pip vim cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# --- Clone IsaacLab pinned to 2.2.1 ---
RUN git clone --depth=1 --branch release/2.2.0 https://github.com/isaac-sim/IsaacLab.git /IsaacLab

# Env vars for IsaacLab
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab
ENV PYTHONPATH=/IsaacLab/source:${PYTHONPATH}

WORKDIR /IsaacLab
RUN ln -s ${ISAACSIM_PATH} _isaac_sim

# Install IsaacLab deps
RUN --mount=type=cache,target=/root/.cache/pip TERM=xterm ./isaaclab.sh --install

# =========================
# Project Integration
# =========================

# Copy project (your BittleHRL repo)
COPY . /BittleHRL

# Install BittleHRL in editable mode
WORKDIR /BittleHRL/source
RUN pip install -e BittleHRL

# Entrypoint
COPY ./docker-deployments/entrypoint.sh /BittleHRL/docker-deployments/entrypoint.sh
RUN chmod +x /BittleHRL/docker-deployments/entrypoint.sh

# Default working directory
WORKDIR /BittleHRL

# PYTHONPATH includes your project
ENV PYTHONPATH=/BittleHRL/source/BittleHRL:/IsaacLab/source:${PYTHONPATH}

ENTRYPOINT ["/BittleHRL/docker-deployments/entrypoint.sh"]
