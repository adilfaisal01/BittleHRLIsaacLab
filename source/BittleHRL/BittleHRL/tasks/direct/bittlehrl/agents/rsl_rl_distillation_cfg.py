
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import  RslRlOnPolicyRunnerCfg

from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherRecurrentCfg

@configclass
class BittleDistillationCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env=32
    max_iterations=150
    save_interval=100
    experiment_name='bittle_test_5'
    logger='tensorboard'

    load_run = "run1"   # your teacher's run folder
    load_checkpoint = "model_1999.pt" 
    
    policy=RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type='scalar',
        student_hidden_dims=[128,128],
        teacher_hidden_dims=[512,512,512],
        activation="relu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False
        ) #same as the aize of the actor network
    
    algorithm=RslRlDistillationAlgorithmCfg(
        learning_rate=1e-4,
        num_learning_epochs=4,
        gradient_length=16

    )

    # or "" for latest