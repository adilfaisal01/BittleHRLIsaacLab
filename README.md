# BittleHRL — CPG-Guided RL for Quadruped Navigation in Isaac Lab

Hierarchical locomotion and goal-directed navigation for the **Petoi Bittle** quadruped, trained entirely in simulation using NVIDIA Isaac Lab.

The key idea: rather than having RL directly output joint angles, a high-level PPO policy outputs **gait parameters** that drive a **Hopf oscillator CPG**, which generates joint trajectories via inverse kinematics. This decouples *how to navigate* (RL's job) from *how to walk* (the CPG's job), producing more stable and transferable locomotion.

---

## Architecture

```
Observation (proprioceptive)
        │
        ▼
  ┌─────────────┐        7 gait params        ┌───────────────────────┐
  │  PPO Policy │ ────────────────────────── ▶ │  Hopf Oscillator CPG  │
  │  (Actor)    │  fwd_vel, period, duty,      │  (Vectorized, batched)│
  └─────────────┘  swing_H, yaw, COM, height   └──────────┬────────────┘
                                                           │ phase signals
                                                           ▼
                                               ┌───────────────────────┐
                                               │   Inverse Kinematics  │
                                               │  hip + knee targets   │
                                               └──────────┬────────────┘
                                                           │
                                                           ▼
                                                  PD Controller → Robot
```

**Training pipeline (PPO → Distillation):**
- **Teacher**: Privileged MLP (512×3 actor / 1024×4 critic) with access to full state including global position
- **Student**: GRU-based recurrent policy (128×2 + 256-dim hidden state) trained via behavior distillation — runs on proprioception only, no GPS

---

## Key Results

| Metric | Value |
|---|---|
| Navigation success rate | **69%** |
| Tip-over rate | **< 1%** |

---

## Method Details

**Reward structure** — two-timescale design:
- *Micro-rewards* (every physics step): upright bonus, torque penalty, joint acceleration penalty, pitch/roll balance terms
- *Macro-rewards* (every RL step): velocity-goal alignment, heading alignment, goal arrival (+20), tip-over penalty

**Domain randomization** (per reset):
- Surface friction: μ ∈ [0.5, 1.2] static, [0.3, 0.9] dynamic
- Body mass perturbation: ±0.05–0.20 kg on base link

**Curriculum learning**: goal distance scales dynamically between 1–2 m based on a rolling 50-episode success rate buffer. Successful robots teleport to new positions; failed robots retry from where they fell.

**Observation space (52-dim)**: gait commands, angular velocity, projected gravity, sin/cos joint positions, joint velocities, CPG phase (normalized), relative goal vector in body frame

---

## Setup

Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then:

```bash
git clone https://github.com/adilfaisal01/BittleHRLIsaacLab.git
cd BittleHRLIsaacLab
python -m pip install -e source/BittleHRL
```

**Train (PPO teacher):**
```bash
python scripts/rsl_rl/train.py --task=Template-Bittlehrl-Direct-v0
```

**Distill to GRU student:**
```bash
python scripts/rsl_rl/play.py --task=Template-Bittlehrl-Distillation-v0
```

**Play:**
```bash
python scripts/rsl_rl/play.py --task=Template-Bittlehrl-Direct-v0
```

Docker deployment configs are available in `docker-deployments/`.

---

## Stack

- **Sim**: NVIDIA Isaac Lab (Isaac Sim backend)
- **RL**: PPO via RSL-RL, with adaptive KL scheduling
- **Distillation**: Teacher→Student via `RslRlDistillationStudentTeacherRecurrentCfg`
- **Robot**: Petoi Bittle (8-DOF quadruped, URDF→USD)
- **Language**: Python, PyTorch

---

## Related

- [SE952--Bittle-Quadruped](https://github.com/adilfaisal01/SE952--Bittle-Quadruped) — earlier gait abstraction work on the same platform
- [NeuroDynamics](https://github.com/adilfaisal01/NeuroDynamics) — transformer-based chaotic system identification
