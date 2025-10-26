FROM nvcr.io/nvidia/isaac-sim:5.0.0

ENV DOCKER_USER_HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

# Work from root
WORKDIR /

# Install utilities
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git python3 python3-pip vim cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip 

# --- Clone IsaacLab (deps layer, changes rarely) ---
RUN git clone --depth=1 https://github.com/isaac-sim/IsaacLab.git 

# Environment variables
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab
ENV PYTHONPATH=${ISAACLAB_PATH}/source:${PYTHONPATH}

WORKDIR /IsaacLab

# --- Install IsaacLab dependencies ---
# --- Link isaac sim ---
RUN ln -s ${ISAACSIM_PATH} _isaac_sim

# Use BuildKit cache for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    TERM=xterm ./isaaclab.sh --install

# ==========================
# === External Project Integration ==
# ==========================

# --- Copy your external Isaac Lab project ---
COPY . /BittleHRL/

# Move into project directory
WORKDIR /BittleHRL


# --- Configure environment ---
# Add Isaac Lab to Python path (we cloned it earlier)
ENV PYTHONPATH="/IsaacLab/source:${PYTHONPATH}"

# --- Install your project independently in editable mode ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e /BittleHRL/source/BittleHRL

# --- Optional: install dev tools like pre-commit ---
RUN pip install pre-commit

# --- Entrypoint setup ---
COPY ./docker-deployments/entrypoint.sh /BittleHRL/docker/entrypoint.sh
RUN chmod +x /BittleHRL/docker/entrypoint.sh

# --- Default working directory ---
WORKDIR /BittleHRL

ENTRYPOINT ["/BittleHRL/docker-deployments/entrypoint.sh"]


