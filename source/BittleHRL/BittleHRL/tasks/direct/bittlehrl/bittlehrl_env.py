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
from isaaclab.terrains import TerrainImporter
import isaaclab.utils.math as math 

## todo: 1. add previous actions to obs space, add smoothness, 2. add 

class BittlehrlEnv(DirectRLEnv):
    cfg: BittlehrlEnvCfg

    def __init__(self, cfg: BittlehrlEnvCfg, render_mode: str | None = None, **kwargs):
        self._actuated_id = 0

        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids = list(range(len(self.robot.data.joint_pos)))
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        #self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        # action_scale = np.array([150,0.667,0.46],dtype=np.float32) #forward velocity, period, duty cycle, swing height, yaw_rate, xCOM_shift, robot_height respectively
        # action_bias=np.array([50,0.33,0.5],dtype=np.float32)

        self.act_scale=torch.tensor([400,0.667,0.46,3,2,30,30],dtype=torch.float32,device=self.device) #action scale
        self.act_bias=torch.tensor([100,0.33,0.5,5,-1,-15,10],dtype=torch.float32,device=self.device) #action bias


        ## residual angles, removing these for now
        # self.jointcorrectionsfactor=torch.deg2rad(torch.tensor(5,device=self.device)) #+/-5 degrees correction, tiny corrections on top of the cpg output
        # self.jointcorrs=torch.rand(self.scene.num_envs,8,device=self.device)

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

        self.gaitcommands=tensorgaitParams(H=torch.full((self.scene.num_envs,),5.678),x_COMshift=torch.full((self.scene.num_envs,),10),robotheight=torch.full((self.scene.num_envs,),20),
                                       yaw_rate=torch.zeros(self.scene.num_envs),
                                       forwardvel=torch.full((self.scene.num_envs,),0),
                                       dutycycle=torch.full((self.scene.num_envs,),0.0),
                                       T=torch.full((self.scene.num_envs,),0.0))
        
        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands)
        self.Q = torch.zeros(self.scene.num_envs, 8,device=self.device)

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

# action continuity
        # In your __init__
        self.actions = torch.zeros((self.scene.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.scene.num_envs, self.cfg.action_space), device=self.device)

# point sampling #

    def sample_points(self, env_ids: torch.Tensor, z_offset: float = 1) -> torch.Tensor:
        origins = self.robot.data.root_link_pos_w
        origins = origins[env_ids]  # Only select the envs we care about
        half_size = self.cfg.scene.env_spacing / 2.0
        margin = 0.5
        
        x_origin = origins[:, 0]
        x_min=x_origin+0.75
        x_max=x_origin+5
        x=sample_uniform(x_min, x_max, (len(env_ids),), device=self.device)
        y_min = origins[:, 1] +0.75
        y_max = origins[:, 1] + 5
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
        super()._setup_scene()
        
        self.robot = Articulation(self.cfg.robot_cfg) #adding the robot
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        terrain=TerrainImporter(self.cfg.terrain)
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
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()
        # decoding normalized actions into convertable values and updating the dataclass
        gait_param_NNs=actions[:,:7]
        actions_clamped_01=torch.sigmoid(gait_param_NNs)
        actions_true=actions_clamped_01*self.act_scale.unsqueeze(0)+self.act_bias.unsqueeze(0)
        fv=actions_true[:,0]
        gaitT=actions_true[:,1]
        dc=actions_true[:,2]
        swingH=actions_true[:,3]
        yaw=actions_true[:,4]
        xCOM=actions_true[:,5]
        r_height=actions_true[:,6]
        self.gaitcommands.forwardvel=torch.clamp(fv,min=100,max=400)
        self.gaitcommands.T=torch.clamp(gaitT,min=0.33,max=1)
        self.gaitcommands.dutycycle=torch.clamp(dc,min=0.5,max=0.9)
        self.gaitcommands.H=torch.clamp(swingH,min=5,max=8)
        self.gaitcommands.yaw_rate=torch.clamp(yaw,min=-2,max=4)
        self.gaitcommands.x_COMshift=torch.clamp(xCOM, min=-15, max=30)
        self.gaitcommands.robotheight=torch.clamp(r_height,min=10,max=30)

        ### adding residuals to joint angles
        # residuals_NN=torch.tanh(actions[:,7:])
        # self.jointcorrs=self.jointcorrectionsfactor*residuals_NN
        # print(f'forwardvel :{fv}',f'T: {gaitT}',f'duty cycle: {dc}')
        # print(f'Normalized actions:{actions}, Clamped actions={actions_clamped}')
        self.hopfoscillator=VectorizedHopfOscillator(gait_pattern=self.gaitcommands) #update the hopf oscillator as part of the caching
        # print(f'true actions: {actions_true}, residuals={self.jointcorrs}')
       
## actual action being processed and applied to the simulated robot

    def _apply_action(self) -> None:
        hip_angle,knee_angle=self._computingjointtragets()
        # print("Q ", self.Q)
        #print("Hip Angle ", hip_angle,"\n","Knee Angle ",knee_angle, "\n")
        self.joint_targets=torch.stack((hip_angle,knee_angle),dim=2)
        self.joint_targets=self.joint_targets.view(self.scene.num_envs,-1)
        self.joint_targets = self.joint_targets * self.joint_signs
        # print("joint target ", self.joint_targets)
        
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=None) #apply action to relevant joints and the PD controller is included (fix needed)

        # print(f'Actual joint position: {self.joint_pos}')
        # print(f'Robot Positions (m): {self.robot.data.root_link_pos_w}')

        sum_torques=torch.sum(torch.square(self.robot.data.applied_torque*self.robot.data.joint_vel),dim=1)
        height_rob=self.robot.data.root_link_pos_w[:,2]
        joint_accel=torch.sum(torch.square(self.robot.data.joint_acc),dim=1) #joint accelerations
        gravity_vector=self.robot.data.projected_gravity_b #gravity vector
        tilt_error=torch.sum(torch.square(gravity_vector[:,:2]),dim=1)


        # balance locomotion of the robot
        torques_sq=torch.square(self.robot.data.applied_torque)
        front_work = torques_sq[:, [1, 5, 3, 7]].sum(dim=-1)
        # Back legs: LB (0, 4) and RB (2, 6)
        back_work = torques_sq[:, [0, 4, 2, 6]].sum(dim=-1)

        # Left legs: LB (0, 4) and LF (1, 5)
        left_work = torques_sq[:, [0, 4, 1, 5]].sum(dim=-1)
        # Right legs: RB (2, 6) and RF (3, 7)
        right_work = torques_sq[:, [2, 6, 3, 7]].sum(dim=-1)

        # 3. Calculate Balances
        # We want these differences to be near zero
        pitch_imbalance = torch.abs(front_work - back_work)
        roll_imbalance = torch.abs(left_work - right_work)

        # 4. Final Balance Reward
        rew_balance = (pitch_imbalance * self.cfg.rew_pitch_scale + 
                        roll_imbalance * self.cfg.rew_roll_scale)
        
        penalties=(
                    self.cfg.rew_tilt*tilt_error+
                    self.cfg.rew_torques*sum_torques+
                    self.cfg.rew_jointaccel*joint_accel+
                    rew_balance
        )
        #per action step, the low level rewards are added
        self.microrewards=self.cfg.upright_reward+penalties+self.microrewards
       
    def _get_observations(self) -> dict:
        
        linear_velocity=self.robot.data.root_lin_vel_b
        x_vel,y_vel,z_vel=torch.unbind(linear_velocity,dim=1) #x,y,z velocities
        gravity_vector=self.robot.data.projected_gravity_b #gravity vector
        tilt_rate=self.robot.data.root_ang_vel_b
        quat = self.robot.data.root_link_quat_w

        pos=self.robot.data.root_link_pos_w #position of the robot in the world, where is the robot in the world
        commands = torch.stack((
        self.gaitcommands.forwardvel,
        self.gaitcommands.T,
        self.gaitcommands.dutycycle,
        self.gaitcommands.H,
        self.gaitcommands.yaw_rate,
        self.gaitcommands.x_COMshift,
        self.gaitcommands.robotheight
    ), dim=-1).to(self.device)          # previous gait commands processed into a tensor
        
        rel_distance_from_goal = self.goal_points- pos
        dist_b=math.quat_apply_inverse(quat,rel_distance_from_goal)
        phase_rad=torch.atan2(self.Q[:,0::2],self.Q[:,1::2])
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float32, device=self.device)
        phase_norm = ((phase_rad + pi_tensor) / (2 * pi_tensor)).float()

        obs = torch.cat(
            (
                pos[:,2].unsqueeze(-1),
                x_vel.unsqueeze(-1),
                y_vel.unsqueeze(-1),
                z_vel.unsqueeze(-1),
                gravity_vector,
                tilt_rate,
                self.joint_vel,
                torch.sin(self.joint_pos),
                torch.cos(self.joint_pos), 
                commands,
                dist_b,
                phase_norm    
            ),
            dim=-1,
        )
        obs=torch.nan_to_num(obs)
        observations = {"policy": obs} 
        #print(f'obs={observations}')
        return observations


    def _get_rewards(self) -> torch.Tensor:
        
        #getting positional and movement data from the robot
        pos = self.robot.data.root_link_pos_w
        quat = self.robot.data.root_link_quat_w
        # roll, pitch, yaw = self._extract_euler_angles()
        linear_velocity_w=self.robot.data.root_lin_vel_w
        # absolute distance from goal, for sparse rewards
        distance_from_goal = torch.norm(self.goal_points[:, :2] - pos[:, :2], dim=-1)
        distance_covered=self.prev_distance-distance_from_goal
        self.prev_distance = distance_from_goal

        # navigation using vectors
        target_vec = self.goal_points- pos #world coordinates
        unit_target_vec=torch.nn.functional.normalize(target_vec,p=2.0,dim=-1)
        linear_velocity_world=linear_velocity_w
        vel_alignment=torch.sum(linear_velocity_world*unit_target_vec,dim=-1) #movement dot product

        # finding the rotation matrix of the robot and if it's facing the goal
        rotation_matrix=math.matrix_from_quat(quat) # output shape is (:,3,3) with the first dim representing batch number
        v_right_w = rotation_matrix[:, :, 0] #right/left
        v_nose_w  = rotation_matrix[:, :, 1] #forward/backward
        v_up_w    = rotation_matrix[:, :, 2] #up/down
        
        cos_nose_to_target = torch.sum(v_nose_w * unit_target_vec, dim=-1) #goal direction vs robot's local y-axis

        #gravity vector
        gravity_vector=self.robot.data.projected_gravity_b #gravity vector
        tilt_error=torch.sum(torch.square(gravity_vector[:,:2]),dim=1)
       
        _,y_vel,_=torch.unbind(linear_velocity_w,dim=1) #x,y,z velocities

        is_static=torch.abs(y_vel)<0.10
        static_pun=is_static.float()*self.cfg.rew_static

        
        ## checking conditional goals and failures
        at_goal = (distance_from_goal < 0.20) & (tilt_error<0.25)
        is_tipped = tilt_error>0.20
        near_goal = (distance_from_goal < 0.50) & (distance_from_goal >= 0.20)

       
        goal_arrival_bots = torch.where(
            at_goal, torch.tensor(self.cfg.goal_reward, device=self.device), torch.tensor(0.0, device=self.device)
        )
        tipped_bots = torch.where(
            is_tipped, torch.tensor(self.cfg.tipped_penalty, device=self.device), torch.tensor(0.0, device=self.device)
        )
        near_goal_bots = torch.where(
            near_goal, torch.tensor(self.cfg.near_goal_reward, device=self.device), torch.tensor(0.0, device=self.device)
        )

        ## contuinuity between residuals to prevent snaps of the robot
        curr_res_angles = torch.sigmoid(self.actions[:, :])
        prev_res_angles = torch.sigmoid(self.prev_actions[:, :])
        # This measures the jump in commanded angle (0.0 to 2.0 range per joint)
        res_continuity = torch.sum(torch.square(curr_res_angles - prev_res_angles), dim=1)

        is_alive=torch.exp(-tilt_error/0.25) #smoothening the binary alive condition

        ### Balance check
        torques_sq=torch.square(self.robot.data.applied_torque)
        front_work = torques_sq[:, [1, 5, 3, 7]].sum(dim=-1)
        # Back legs: LB (0, 4) and RB (2, 6)
        back_work = torques_sq[:, [0, 4, 2, 6]].sum(dim=-1)

        # Left legs: LB (0, 4) and LF (1, 5)
        left_work = torques_sq[:, [0, 4, 1, 5]].sum(dim=-1)
        # Right legs: RB (2, 6) and RF (3, 7)
        right_work = torques_sq[:, [2, 6, 3, 7]].sum(dim=-1)

        # 3. Calculate Balances
        # We want these differences to be near zero
        pitch_imbalance = torch.abs(front_work - back_work)
        roll_imbalance = torch.abs(left_work - right_work)
        
        #print(f'microrewards={self.microrewards},distance={distance_covered},goal_arrival_bots={goal_arrival_bots},rew_distance={distance_covered* self.cfg.rew_dist_goal}')
        reward = (
            vel_alignment* self.cfg.rew_dist_goal*is_alive*torch.clamp(cos_nose_to_target,min=0) +
            goal_arrival_bots/10 +
            tipped_bots/10 +
            near_goal_bots/10+
            4*torch.tanh(self.microrewards/self.cfg.decimation)+
            self.cfg.rew_action_continuity * res_continuity+
            self.cfg.rew_ep_len*self.episode_length_buf+
            self.cfg.rew_heading*cos_nose_to_target+
            static_pun
            
        )
        reward_components = {
        "rew_distance": (vel_alignment * self.cfg.rew_dist_goal*is_alive*torch.clamp(cos_nose_to_target,min=0)).mean().detach(),
        "rew_continuity": (self.cfg.rew_action_continuity * res_continuity).mean().detach(),
        "rew_micro_penalties": (4*torch.tanh(self.microrewards/self.cfg.decimation)).mean().detach(),
        "rew_total": reward.mean().detach(),
        "direction": (self.cfg.rew_heading*cos_nose_to_target).mean().detach(),
        "tip_rate":is_tipped.float().mean().detach(),
        "success_rate" : (at_goal.float().mean()*100).detach(),
        "balance_pitch": pitch_imbalance.mean().detach(),
        "balance_roll": roll_imbalance.mean().detach(),
        "total_work": torques_sq.sum(dim=-1).mean().detach(),
        "near_goal": (near_goal_bots.float().mean()*100).detach()  
    }

    # 2. Store in the extras dictionary (Isaac Lab DirectRLEnv convention)
        if not hasattr(self, "extras"):
            self.extras = {}
        
        # RSL-RL specifically looks for the 'episode' key to log scalars
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        
        for key, value in reward_components.items():
            self.extras["episode"][key] = value
        # print(f'microrewards={self.microrewards}')
        # print(f'distance_R={distance_covered* self.cfg.rew_dist_goal}')
        # print(f'final_rewards={reward}')
        reward=torch.clamp(reward,min=-1000,max=1000)
        reward=torch.nan_to_num(reward,nan=0)
        self.microrewards=torch.zeros(self.scene.num_envs,device=self.device)  #each RL steps all the low level rewards are set to zero, so the microrewards restart accumulation
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        pos=self.robot.data.root_link_pos_w

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        gravity_vector=self.robot.data.projected_gravity_b #gravity vector
        tilt_error=torch.sum(torch.square(gravity_vector[:,:2]),dim=1)
        
        dist = torch.norm(self.goal_points[:, :2] - pos[:, :2], dim=-1)        
        success = (dist< 0.20) & (tilt_error<0.25)#find the successful robots, if succeeded, no need to continue
        absolute_tipover=tilt_error>0.65  #if the robot tips over by 90 degree, end it

        # if success.any():
        #     env_ids=torch.nonzero(success).squeeze(-1)
        #     self.sample_goals(env_ids=torch.nonzero(success).squeeze(-1))
        #     print(f'yeee boii, {env_ids[success]}')


        # print(f'TO={time_out}, SUC={success}, AT={absolute_tipover}')

        return  absolute_tipover | success, time_out

    def  _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Check which envs succeeded
        pos = self.robot.data.root_link_pos_w[env_ids]
        dist = torch.norm(self.goal_points[env_ids, :2] - pos[:, :2], dim=-1)
        gravity_vector=self.robot.data.projected_gravity_b #gravity vector
        tilt_error=torch.sum(torch.square(gravity_vector[:,:2]),dim=1)
        success = (dist < 0.20) & (torch.abs(tilt_error[env_ids])<0.25) #create mask here for success, filtering
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
            self.sample_goals(env_ids=succ_ids)
             # Save updated spawn state for next reset
            self.spawn_root_states[succ_ids] = root_state
            self.spawn_joint_pos[succ_ids] = joint_pos
            self.spawn_joint_vel[succ_ids] = joint_vel

        # Reapply cached spawn state (works for both success & fail cases)
        root_state = self.spawn_root_states[env_ids]
        joint_pos = self.spawn_joint_pos[env_ids]
        joint_vel = self.spawn_joint_vel[env_ids]
        self.prev_distance[env_ids] = torch.norm(self.goal_points[env_ids, :2] - root_state[:, :2], dim=-1)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


