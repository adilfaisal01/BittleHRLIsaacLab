# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import math
from math import ceil, sqrt
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.sensors.frame_transformer import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg,ImuCfg
import gymnasium.spaces as spaces
import torch 
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp.events as mdp
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg,GaussianNoiseCfg,NoiseModelCfg

BITTLE_ASSET_DIR = Path(__file__).resolve().parent
## Add domain randomization code here. situations to be simulated: 1. pushes 2. robot material per reset
@configclass
class EventCfg:
   robot_physics_material=EventTerm(
       func=mdp.randomize_rigid_body_material,
       mode="reset",
       params={
           "asset_cfg": SceneEntityCfg("robot",body_names=".*"),
           "static_friction_range":(0.5,1.2),
           "dynamic_friction_range":(0.3,0.9),
           "restitution_range":(0.0,0.0),
           "num_buckets":256,
       }
   )
   add_base_mass=EventTerm(
       func=mdp.randomize_rigid_body_mass,
       mode="reset",
       params={
           "asset_cfg": SceneEntityCfg("robot",body_names="base_frame_link"),
           "mass_distribution_params":(-0.05,0.20),
           "operation":"add",
       }


   )
    



@configclass
class BittlehrlEnvCfg(DirectRLEnvCfg):
    # ====== ENV / TIMING ======
    decimation = 40 #number of control steps between policy updates, policy runs at 5 Hz, simulation at 100 Hz
    episode_length_s = 40
    action_space = spaces.Box(low= 0,high=1,dtype=np.float32,shape=(13,)) #normalized actions
    # 1) your basic scalar limits
    
    observation_space = 25

    state_space = 0

    ## Noise model-- adding gaussian noise to action and observations
        # only per-step noise
    action_noise_model = NoiseModelCfg(noise_cfg=GaussianNoiseCfg(mean=0.0,std=0.01,operation="add"))

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model= NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
    
    # ====== SIMULATION ======
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,
        render_interval=1,
        device="cuda",
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=15.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # ====== ROBOT ======
    bittle = ArticulationCfg(
        prim_path="/World/envs/env_.*/bittle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BITTLE_ASSET_DIR}/Bittle_URDF/bittle/bittle.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={
                "left_back_shoulder_joint" : 0.0, 
                "left_front_shoulder_joint" : 0.0, 
                "right_back_shoulder_joint" : 0.0, 
                "right_front_shoulder_joint" : 0.0, 
                "left_back_knee_joint" : 0.0, 
                "left_front_knee_joint" : 0.0, 
                "right_back_knee_joint" : 0.0, 
                "right_front_knee_joint" : 0.0
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body_pd": ImplicitActuatorCfg(
                joint_names_expr=["left_.*", "right_.*"],
                effort_limit_sim=120.0,
                velocity_limit_sim=20.0,
                stiffness=20.0,
                damping=10.0,
            ),
        },
    )
    robot_cfg: ArticulationCfg = bittle
    events: EventCfg = EventCfg()

    #sensors

    # imu=ImuCfg(
    #     prim_path="/World/envs/env_.*/bittle/base_frame_link/mainboard_link/imu_link/IMU_Sensor",debug_vis=True)


    # ====== FRAME TRANSFORMER (for feet) ======
    # lf_rf_transformer: FrameTransformerCfg = FrameTransformerCfg(
    #     prim_path="/World/envs/env_.*/bittle/pelvis",
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="/World/envs/env_.*/bittle/leftFoot",
    #             name="leftFoot"
    #         ),
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="/World/envs/env_.*/bittle/rightFoot",
    #             name="rightFoot"
    #         ),
    #     ],
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    # )

    # ====== REWARD WEIGHTS (from GymWrapper) =====
    
    # micro rewrd terms, every action cycle these rewards are taken and measured
    rew_torques=-0.01
    rew_roll=-0.009
    rew_pitch=-0.009
    rew_pitchrate=-0.0007
    rew_rollrate=-0.0007
    # macro rewards, collected every RL step
    rew_dist_goal=10
    goal_reward=10
    tipped_penalty=-1
    near_goal_reward=9
    upright_reward=0.40
    

    # ====== RAY CASTER (pelvis → ground) ======
    # ray_caster: RayCasterCfg = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/bittle/torso",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.05)),
    #     mesh_prim_paths=["/World/Ground"],
    #     max_distance=2.0,
    #     pattern_cfg=patterns.GridPatternCfg(
    #         size=(0.0, 0.0),
    #         resolution=1,
    #         direction=(0, 0.0, -1.0),
    #     ),
    #     debug_vis=True,
    # )

    # ====== TERRAIN (generator) ======

    N = int(scene.num_envs)
    S = float(scene.env_spacing)
    rows = int(ceil(sqrt(N)))
    cols = int(ceil(N / rows))
    static_fric=float(torch.normal(0.5,0.1,size=(1,1)))
    dyn_fric= float(torch.normal(0.3,0.15,size=(1,1)))

    if static_fric>dyn_fric:
        pass
    else:
        dyn_fric=static_fric-float(torch.normal(0.05,0.014,size=(1,1)))

    generator = TerrainGeneratorCfg(
        size=(S, S),
        border_width=0.1,
        border_height=-1.0,
        num_rows=rows,
        num_cols=cols,
        color_scheme="height",
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
            #     # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            #     #     proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            #     # ),
            #     # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            #     #     proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            #     # ),
            #     # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            #     #     proportion=0.05, step_height_range=(0.0, 0.1), step_width=0.3,
            #     #     platform_width=3.0, border_width=1.0, holes=False
            #     # ),
            #     # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            #     #     proportion=0.05, step_height_range=(0.0, 0.1), step_width=0.3,
            #     #     platform_width=3.0, border_width=1.0, holes=False
            #     # ),
            # # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            # #    proportion=0.33, amplitude_range=(0.0, 0.06), num_waves=4, border_width=0.25
            # #),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.7, noise_range=(0.005, 0.01), noise_step=0.005, border_width=0.25)
            })
    terrain = TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="generator",
        terrain_generator=generator,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=static_fric,
                dynamic_friction=dyn_fric,
            ),
            debug_vis=False,
        )

    terrain_cfg: TerrainImporterCfg=terrain
