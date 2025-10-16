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
from .inversegait import JointOffsets, hiplength,kneelength
from .vectorizedBittle_Locomotion import tensor_connection_weight_matrix_R,tensorgaitParams,VectorizedHopfOscillator,VectorizedMotionPlanning
import numpy as np

## todo: 1. add previous actions to obs space, add smoothness, 2. add 

class BittlehrlEnv(DirectRLEnv):
    cfg: BittlehrlEnvCfg

    def __init__(self, cfg: BittlehrlEnvCfg, render_mode: str | None = None, **kwargs):
        self._actuated_id = 0

        super().__init__(cfg, render_mode, **kwargs)
        
        self.joint_ids = list(range(len(self.robot.data.joint_pos)))
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        # self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        # action_scale = np.array([150,0.667,0.46],dtype=np.float32) #forward velocity, period, duty cycle respectively
        # action_bias=np.array([50,0.33,0.5],dtype=np.float32)

        self.act_scale=torch.tensor([250,0.667,0.46],dtype=torch.float32,device=self.device) #action scale
        self.act_bias=torch.tensor([100,0.33,0.5],dtype=torch.float32,device=self.device) #action bias

        # per-env goal points (xyz)
        self.goal_points = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.sample_goals(first_time=True)

        # reward bookkeeping
        self.prev_distance = torch.zeros(self.scene.num_envs, device=self.device)

        # spawn state tensors (persistent across resets)
        self.spawn_root_states = self.robot.data.default_root_state.clone()
        self.spawn_joint_pos = self.robot.data.default_joint_pos.clone()
        self.spawn_joint_vel = self.robot.data.default_joint_vel.clone()

        self.first_reset = True

        self.gaitcommands=tensorgaitParams(H=torch.full((self.scene.num_envs,),5.678),x_COMshift=torch.full((self.scene.num_envs,),-10),robotheight=torch.full((self.scene.num_envs,),20),
                                       yaw_rate=torch.zeros(self.scene.num_envs),
                                       forwardvel=torch.full((self.scene.num_envs,),0),
                                       dutycycle=torch.full((self.scene.num_envs,),0.0),
                                       T=torch.full((self.scene.num_envs,),0.0))
        
        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands)
        self.Q = torch.zeros(self.scene.num_envs, 8)

        # Initialize Hopf oscillator phases
        self.trot_phase_difference = torch.tensor([0.496, 0, 0, 0.496], dtype=torch.float32) * 2*torch.pi
        for i in range(4):
            self.Q[:, 2*i] = torch.cos(self.trot_phase_difference[i])
            self.Q[:, 2*i+1] = torch.sin(self.trot_phase_difference[i])

        self.R_trot = tensor_connection_weight_matrix_R(self.trot_phase_difference)
        self.joint_index_map =   { 
                                "Right Front": [3, 7],
                                "Left Front": [1, 5],
                                "Right Back": [2, 6],
                                "Left Back": [0, 4],
                            }
        
        self.microrewards=torch.zeros(self.scene.num_envs,device=self.device) 

        self.joint_sign_map = {
                "right_front_shoulder_joint": -1,
                "right_front_knee_joint": -1,
                "left_front_shoulder_joint": 1,
                "left_front_knee_joint": 1,
                "right_back_shoulder_joint": -1,
                "right_back_knee_joint": -1,
                "left_back_shoulder_joint": 1,
                "left_back_knee_joint": 1
            }
        
        joint_names = [
        "left_back_shoulder_joint", 
        "left_front_shoulder_joint", 
        "right_back_shoulder_joint", 
        "right_front_shoulder_joint", 
        "left_back_knee_joint", 
        "left_front_knee_joint", 
        "right_back_knee_joint", 
        "right_front_knee_joint"
        ]

# Create sign multiplier based on joint names
        self.joint_signs = torch.tensor([self.joint_sign_map[name] for name in joint_names],device=self.device) 

# point sampling #

    def sample_points(self, env_ids: torch.Tensor, z_offset: float = 1) -> torch.Tensor:
        origins = self.robot.data.root_link_pos_w
        origins = origins[env_ids]  # Only select the envs we care about
        half_size = self.cfg.scene.env_spacing / 2.0
        margin = 0.5

        x = origins[:, 0]
        y_min = origins[:, 1] +0.75
        y_max = origins[:, 1] + 1.5
        y = sample_uniform(y_min, y_max, (len(env_ids),), device=self.device)
        z = origins[:, 2] + z_offset
        # print(f'GP: x={x}, y={y}, z={z}')

        return torch.stack([x, y, z], dim=-1)

    # === Goal Sampling ===
    def sample_goals(self, env_ids: torch.Tensor | None = None, first_time=False):
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)

        self.goal_points[env_ids] = self.sample_points(env_ids)
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
        # self.imu=Imu(self.cfg.imu)
        # self.scene.sensors["Imu"]=self.imu

## physics hlpers for custom locomotion engine

    def _cpg_update(self):
        self.Q=self.hopfoscillator.tensor_hopf_cpg_dot(self.Q,R=self.R_trot,delta=0.01,b=0.50,mu=1,alpha=1,gamma=1,dt=self.cfg.sim.dt)
        return self.Q
    
    def _computingjointtragets(self):
        self.Q=self._cpg_update() #update the cpg signal being sent to the quadruped
        self.motionplanner=VectorizedMotionPlanning(gait_pattern=self.gaitcommands,JointOffsets=JointOffsets,L1=hiplength,L2=kneelength,z_rest_foot=-68.92) #instantiate the class
        xhopf=self.Q[:,0::2]
        zhopf=self.Q[:,1::2]
        x_traj,z_traj=self.motionplanner.tensor_TrajectoryGenerator(xhopf,zhopf)
        hip_angle,knee_angle=self.motionplanner.tensor_InverseKinematics(x_traj,z_traj)

        return hip_angle,knee_angle #size (num_envs,8)
## action step,s in pre-physics step, the actions dictated by the RL are cached
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        
        # decoding normalized actions into convertable values and updating the dataclass
        actions_clamped=torch.sigmoid(actions)
        actions_true=actions_clamped*self.act_scale.unsqueeze(0)+self.act_bias.unsqueeze(0)
        fv=actions_true[:,0]
        gaitT=actions_true[:,1]
        dc=actions_true[:,2]
        self.gaitcommands.forwardvel=torch.clamp(fv,min=100,max=250)
        self.gaitcommands.T=torch.clamp(gaitT,min=0.33,max=1)
        self.gaitcommands.dutycycle=torch.clamp(dc,min=0.5,max=0.9)
        # print(f'forwardvel :{fv}',f'T: {gaitT}',f'duty cycle: {dc}')
        # print(f'Normalized actions:{actions}, Clamped actions={actions_clamped}')
        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands) #update the hopf oscillator as part of the caching
        self.microrewards=torch.zeros(self.scene.num_envs,device=self.device)  #each RL steps all the low level rewards are set to zero, so the microrewards restart accumulation
        # print(f'true actions: {actions_true}')
       
## actual action being processed and applied to the simulated robot

    def _apply_action(self) -> None:
        hip_angle,knee_angle=self._computingjointtragets()
        # print("Q ", self.Q)
        # print("Hip Angle ", hip_angle,"\n","Knee Angle ",knee_angle, "\n")
        self.joint_targets=torch.stack((hip_angle,knee_angle),dim=2)
        self.joint_targets=self.joint_targets.view(self.scene.num_envs,-1)
        self.joint_targets = self.joint_targets * self.joint_signs
        # print("joint target ", self.joint_targets)
        
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=None) #apply action to relevant joints and the PD controller is included (fix needed)

        # print(f'Actual joint position: {self.joint_pos}')
        # print(f'Robot Positions (m): {self.robot.data.root_link_pos_w}')

        roll,pitch,_=self._extract_euler_angles() #3 separate angles
        sum_torques=torch.sum(torch.square(self.robot.data.applied_torque),dim=1)
        height_rob=self.robot.data.root_link_pos_w[:,2]

        roll_rate=self.robot.data.root_ang_vel_b[:,0]
        pitch_rate=self.robot.data.root_ang_vel_b[:,1]
        self.microrewards=0.90*self.microrewards+(self.cfg.rew_roll*torch.abs(roll)+
                                                  self.cfg.rew_pitch*torch.abs(pitch)+
                                                  self.cfg.rew_torques*sum_torques+
                                                  self.cfg.rew_height*height_rob+
                                                  self.cfg.rew_rollrate*torch.abs(roll_rate)+
                                                  self.cfg.rew_pitchrate*torch.abs(pitch_rate)+
                                                  self.cfg.rew_height*height_rob) #per action step, the low level rewards are added
        
    def _get_observations(self) -> dict:
        
        linear_velocity=self.robot.data.root_lin_vel_b
        x_vel,y_vel,z_vel=torch.unbind(linear_velocity,dim=1) #x,y,z velocities
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
                self.joint_pos,
            
                
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

# ew_vx=-1
#     rew_vy=+5
#     rew_vz=-1
#     rew_joint_energy=-0.01
#     rew_roll=-0.5
#     rew_pitch=-0.5
#     rew_dist_goal=-4 
#     goal_reward=500
#     tipped_penalty=-500

    def _get_rewards(self) -> torch.Tensor:
        pos=self.robot.data.root_link_pos_w
        # print(f'robot position :{pos}')
        roll,pitch,yaw=self._extract_euler_angles()
        distance_from_goal=torch.abs(self.goal_points[:, 1] - pos[:, 1]) #find the distance from goal, every 5 seconds
        self.prev_distance=distance_from_goal

        at_goal= (distance_from_goal < 0.20) & (torch.abs(roll) < 0.3) & (torch.abs(pitch) < 0.2)

        is_tipped=(torch.abs(roll)>0.8) | (torch.abs(pitch)>0.8)
        near_goal=(distance_from_goal < 0.50) & (distance_from_goal >= 0.20)
        goal_arrival_bots=torch.where(at_goal,torch.tensor(self.cfg.goal_reward,device=self.device),torch.tensor(0.0,device=self.device))
        tipped_bots=torch.where(is_tipped,torch.tensor(self.cfg.tipped_penalty,device=self.device),torch.tensor(0.0,device=self.device))
        near_goal_bots=torch.where(near_goal,torch.tensor(self.cfg.near_goal_reward,device=self.device),torch.tensor(0.0,device=self.device))

        # print(f'distance from goal:{distance_from_goal}')
        # print(f'Microrewards: {self.microrewards}, Near goal reward: {near_goal_bots}, At goal reward: {goal_arrival_bots}')

        reward=(
           distance_from_goal*self.cfg.rew_dist_goal+
            goal_arrival_bots+
            tipped_bots+
            near_goal_bots+
            self.microrewards
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        pos=self.robot.data.root_link_pos_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        roll, pitch,_ = self._extract_euler_angles()
        dist =torch.abs(-pos[:,1]+self.goal_points[:,1]) 
        success = (dist < 0.20) & (torch.abs(roll) < 0.3) & (torch.abs(pitch) < 0.2) #find the successful robots, if succeeded, no need to continue
        absolute_tipover=(torch.abs(roll)>2.00) | (torch.abs(pitch)>2.00) # if the robot tips over by 90 degree, end it

        if success.any():
            self.sample_goals(env_ids=torch.nonzero(success).squeeze(-1))
            print(f'yeee boii, {success}')

        # print(f'TO={time_out}, SUC={success}, AT={absolute_tipover}')

        return  absolute_tipover | success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Check which envs succeeded
        pos = self.robot.data.root_link_pos_w[env_ids]
        roll, pitch,_ = self._extract_euler_angles()
        dist = torch.abs(self.goal_points[env_ids, 1] - pos[:, 1])
        success = (dist < 0.20) & (torch.abs(roll[env_ids]) < 0.3) & (torch.abs(pitch[env_ids]) < 0.2) #create mask here for success, filtering
        succ_ids = env_ids[success]

        if self.first_reset or len(succ_ids) > 0:
            
            if self.first_reset:
                succ_ids = env_ids
                self.first_reset = False
            # Pull defaults for successful envs
            joint_pos = self.robot.data.default_joint_pos[succ_ids]
            joint_vel = self.robot.data.default_joint_vel[succ_ids]
            root_state = self.robot.data.default_root_state[succ_ids].clone()

            # Sample XY from training ground and enforce configured Z height
            spawn_points = self.sample_points(succ_ids, z_offset=1)
            root_state[:, 0:2] = spawn_points[:,0:2]
            root_state[:,2] = 0.1
             # Save updated spawn state for next reset
            self.spawn_root_states[succ_ids] = root_state
            self.spawn_joint_pos[succ_ids] = joint_pos
            self.spawn_joint_vel[succ_ids] = joint_vel

        # Reapply cached spawn state (works for both success & fail cases)
        root_state = self.spawn_root_states[env_ids]
        joint_pos = self.spawn_joint_pos[env_ids]
        joint_vel = self.spawn_joint_vel[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
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


