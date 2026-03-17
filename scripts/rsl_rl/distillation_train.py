import argparse
import os
import sys
from datetime import datetime
import torch
import gymnasium as gym

# Isaac Lab / RSL-RL
from isaaclab.app import AppLauncher
import cli_args  

# 1. CLI SETUP
parser = argparse.ArgumentParser(description="Bittle Hierarchical Distillation.")
parser.add_argument("--teacher_path", type=str, required=True, help="Path to pre-trained expert actor (.pt)")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel environments.")
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU training.")
parser.add_argument("--task", type=str, default="Bittle-HRL-v0")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Launch App
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-Launch Imports
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import BittleHRL.tasks  

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Apply CLI Overrides
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
    
    # Create Env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 2. INITIALIZE RUNNER
    log_dir = os.path.join("logs", "distillation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # 3. LOAD & FREEZE TEACHER
    print(f"[INFO] Loading Teacher: {args_cli.teacher_path}")
    checkpoint = torch.load(args_cli.teacher_path, map_location=agent_cfg.device)
    
    # Handle standard RSL-RL checkpoint format
    teacher_state = checkpoint.get('model_state_dict', checkpoint)
    runner.alg.policy.teacher.load_state_dict(teacher_state)
    
    for param in runner.alg.policy.teacher.parameters():
        param.requires_grad = False
    runner.alg.policy.teacher.eval()

    # 4. RUN DISTILLATION
    # RSL-RL will automatically log 'behavior' (MSE) to TensorBoard
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()