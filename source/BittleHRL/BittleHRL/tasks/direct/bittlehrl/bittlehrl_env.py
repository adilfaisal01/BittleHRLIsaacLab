# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject,RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import Imu

from .bittlehrl_env_cfg import BittlehrlEnvCfg
from inversegait import JointOffsets, hiplength,kneelength
from vectorizedBittle_Locomotion import tensor_connection_weight_matrix_R,tensorgaitParams,VectorizedHopfOscillator,VectorizedMotionPlanning
import numpy as np


class BittlehrlEnv(DirectRLEnv):
    cfg: BittlehrlEnvCfg

    def __init__(self, cfg: BittlehrlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids = list(range(len(self.robot.data.joint_pos)))
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._actuated_ids:torch.Tensor | None
        self.act_scale=torch.tensor(cfg.action_scale,dtype=torch.float32,device=self.device) #action scale
        self.act_bias=torch.tensor(cfg.action_bias,dtype=torch.float32,device=self.device) #action bias

        # per-env goal points (xyz)
        self.goal_points = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.sample_goals(first_time=True)

        # reward bookkeeping
        self.prev_actions = torch.zeros(
            (self.scene.num_envs, len(self.joint_pos)), device=self.device
        )
        self.prev_distance = torch.zeros(self.scene.num_envs, device=self.device)
        self.was_tipped_last = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)

        # spawn state tensors (persistent across resets)
        self.spawn_root_states = self.robot.data.default_root_state.clone()
        self.spawn_joint_pos = self.robot.data.default_joint_pos.clone()
        self.spawn_joint_vel = self.robot.data.default_joint_vel.clone()

        self.first_reset = True

        self.gaitcommands=tensorgaitParams(H=torch.full((self.scene.num_envs,),5.678),x_COMshift=torch.full((self.scene.num_envs,),0),robotheight=torch.full((self.scene.num_envs,),20),
                                       yaw_rate=torch.zeros(self.scene.num_envs),
                                       forwardvel=torch.full((self.scene.num_envs,),50),
                                       dutycycle=torch.full((self.scene.num_envs,),0.0),
                                       T=torch.full((self.scene.num_envs,),0.0))
        
        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands)
        Q = torch.zeros(self.scene.num_envs, 8)

        # Initialize Hopf oscillator phases
        self.trot_phase_difference = torch.tensor([0.496, 0, 0, 0.496], dtype=torch.float32) * 2*torch.pi
        for i in range(4):
            Q[:, 2*i] = torch.cos(self.trot_phase_difference[i])
            Q[:, 2*i+1] = torch.sin(self.trot_phase_difference[i])

        self.R_trot = tensor_connection_weight_matrix_R(self.trot_phase_difference)
        self.time=0 #time of the simulation
        self.highlevelfrequency=0.2 #decision every 5 seconds
        self.HLsteps=int(0.2/0.01)
        self.joint_index_map =   { 
                                "Right Front": [3, 7],
                                "Left Front": [1, 5],
                                "Right Back": [2, 6],
                                "Left Back": [0, 4],
                            }

# point sampling #

    def sample_points(self, env_ids: torch.Tensor, z_offset: float = 2) -> torch.Tensor:
        origins = self.scene.env_origins[env_ids]
        half_size = self.cfg.scene.env_spacing / 2.0
        margin = 0.5

        x = origins[:, 0]
        y_min = origins[:, 1] - half_size + margin
        y_max = origins[:, 1] + half_size - margin

        y = sample_uniform(y_min, y_max, (len(env_ids),), device=self.device)
        z = origins[:, 2] + z_offset

        return torch.stack([x, y, z], dim=-1)

    # === Goal Sampling ===
    def sample_goals(self, env_ids: torch.Tensor | None = None, first_time=False):
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)

        self.goal_points[env_ids] = self.sample_points(env_ids, z_offset=0.5)
        # self._spawn_goal_markers(env_ids)

    def _spawn_goal_markers(self, env_ids: torch.Tensor):
        for i in env_ids.tolist():
            prim_path = f"/World/goals/goal_{i}"
            pos = self.goal_points[i].cpu().numpy().tolist()

            # Create sphere marker config
            sphere_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=[0.0, 1.0, 0.0]),
                rigid_props=None,
            )

            goal_cfg = RigidObjectCfg(prim_path=prim_path, spawn=sphere_cfg)

            if prim_path in self.scene.rigid_objects:
                self.scene.rigid_objects[prim_path].set_world_pose(pos)
            else:
                goal = RigidObject(cfg=goal_cfg)
                self.scene.rigid_objects.add(goal)
                goal.set_world_pose(pos)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        #add IMU
        self.imu=Imu(self.cfg.imu)
        self.scene.sensors["Imu"]=self.imu

    def _cpg_update(self):
        self.Q=self.hopfoscillator.tensor_hopf_cpg_dot(self.Q,R=self.R_trot,delta=0.01,b=0.50,mu=1,alpha=10,gamma=10,dt=self.cfg.sim.dt)
        return self.Q
    
    def _computingjointtragets(self):
        self.Q=self._cpg_update() #update the cpg signal being sent to the quadruped
        self.motionplanner=VectorizedMotionPlanning(gait_pattern=self.gaitcommands,JointOffsets=JointOffsets,L1=hiplength,L2=kneelength,z_rest_foot=-68.92) #instantiate the class
        xhopf=self.Q[:,0::2]
        zhopf=self.Q[:,1::2]
        x_traj,z_traj=self.motionplanner.tensor_TrajectoryGenerator(xhopf,zhopf)
        hip_angle,knee_angle=self.motionplanner.tensor_InverseKinematics(x_traj,z_traj)

        return hip_angle,knee_angle #size (num_envs,8)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        
        # decoding normalized actions into convertable values and updating the dataclass
        actions_true=actions*self.act_scale.unsqueeze(0)+self.act_bias.unsqueeze(0)
        fv=actions_true[:,0]
        gaitT=actions_true[:,1]
        dc=actions_true[:,2]
        self.gaitcommands.forwardvel=fv
        self.gaitcommands.T=gaitT
        self.gaitcommands.dutycycle=dc

        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands) #update the hopf oscillator
        #applying low level control
        self.Q=self._cpg_update()
        hip_angle,knee_angle=self._computingjointtragets()
        self.joint_targets=torch.stack(hip_angle,knee_angle,dim=2)
        self.joint_targets=self.joint_targets.view(self.scene.num_envs,-1)

        return self.joint_targets

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets,joint_ids=self._actuated_ids) #apply action to relevant joints and the PD controller is included (fix needed)
        
    def _get_observations(self) -> dict:
        
        linear_velocity=self.robot.data.root_lin_vel_b
        x_vel,y_vel,z_vel=linear_velocity #x,y,z velocities
        roll,pitch,yaw=self._extract_euler_angles() #3 separate angles
        pos=self.robot.data.root_link_pos_w #position of the robot in the world, where is the robot in the world
        obs = torch.cat(
            (
                pos,
                x_vel.unsqueeze(-1),
                y_vel.unsqueeze(-1),
                z_vel.unsqueeze(-1),
                roll.unsqueeze(-1),
                pitch.unsqueeze(-1),
                yaw.unsqueeze(-1),
                self.joint_vel,
                self.joint_pos
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.s,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _extract_euler_angles(self):
        quat = self.robot.data.root_link_quat_w
        roll = torch.atan2(
            2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
            1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        )
        pitch = torch.asin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
        yaw=torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        return roll, pitch,yaw


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward