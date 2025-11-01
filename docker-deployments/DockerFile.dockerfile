# Base Isaac Sim image
FROM nvcr.io/nvidia/isaac-sim:5.0.0

ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKER_USER_HOME=/root

# Work from root
WORKDIR /

# Install utilities
RUN apt-get update && apt-get install -y \
    git python3 python3-pip vim cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# --- Clone IsaacLab (pick a stable branch) ---
RUN git clone --depth=1 --branch release/2.2.0 https://github.com/isaac-sim/IsaacLab.git

# Environment variables for IsaacLab
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab
ENV PYTHONPATH=${ISAACLAB_PATH}/source:${PYTHONPATH}

WORKDIR /IsaacLab

# Link Isaac Sim
RUN ln -s ${ISAACSIM_PATH} _isaac_sim

# Install IsaacLab dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    TERM=xterm ./isaaclab.sh --install

# --- Copy your external project ---
COPY . /BittleHRL

# Make PYTHONPATH include your project source
ENV PYTHONPATH=/BittleHRL/source:${PYTHONPATH}

WORKDIR /BittleHRL

# Install your project in editable mode
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./source/BittleHRL

# Optional dev tools
RUN pip install pre-commit

# Entrypoint
COPY ./docker-deployments/entrypoint.sh /BittleHRL/docker-deployments/entrypoint.sh
RUN chmod +x /BittleHRL/docker-deployments/entrypoint.sh

WORKDIR /BittleHRL
ENTRYPOINT ["/BittleHRL/docker-deployments/entrypoint.sh"]
