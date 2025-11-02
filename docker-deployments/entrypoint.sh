#!/bin/bash
set -e
set -x

echo "🚀 Launching BittleHRL training..."

# Hardcoded training script and settings, change this to match my environment implementation details
/IsaacLab/isaaclab.sh -p -m torch.distributed.run --nnodes=1 --nproc_per_node=4 /BittleHRL/scripts/skrl/train.py \
  --task Template-Bittlehrl-Direct-v0 \
  --num_envs $NUM_ENVS \
  --max_iterations $MAX_ITERS \
  --video --headless --distributed
