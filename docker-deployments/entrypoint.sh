#!/bin/bash
set -e
set -x

echo "🚀 Launching BittleHRL training..."

# Hardcoded training script and settings, change this to match my environment implementation details
/IsaacLab/isaaclab.sh -p /BittleHRL/scripts/skrl/train.py \
  --task Template-Bittlehrl-Direct-v0 \
  --num_envs ${NUM_ENVS:-2} \
  --max_iters ${MAX_ITERS:-500} \
  --video --headless
