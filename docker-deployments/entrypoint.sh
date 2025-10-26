#!/bin/bash
set -e
set -x

echo "🚀 Launching BittleHRL training..."

# Hardcoded training script and settings, change this to match my environment implementation details
/IsaacLab/isaaclab.sh -p /BittleHRL/scripts/skrl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 64 \
    --headless \
    --video \
    --cpu-only
