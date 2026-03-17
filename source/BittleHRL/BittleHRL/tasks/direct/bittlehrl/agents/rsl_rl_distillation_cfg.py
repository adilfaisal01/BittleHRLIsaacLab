
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import  RslRlOnPolicyRunnerCfg

from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherRecurrentCfg, RslRlDistillationRunnerCfg

@configclass
class BittleDistillationCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env=32
    max_iterations=150
    save_interval=100
    experiment_name='bittle_distillation'
    logger='tensorboard'
    
    policy=RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.01,
        noise_std_type='scalar',
        student_hidden_dims=[256,256],
        teacher_hidden_dims=[512,512,512],
        activation="relu",
        teacher_obs_normalization= True,
        student_obs_normalization= True,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False
        ) #same as the aize of the actor network
    
    algorithm=RslRlDistillationAlgorithmCfg(
        learning_rate=1e-4,
        num_learning_epochs=4,
        loss_type="mse",
        optimizer="adam",
        gradient_length=16

    )
