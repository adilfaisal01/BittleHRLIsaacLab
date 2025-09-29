import torch
from dataclasses import dataclass

# NUCLEAR OPTION: SET GLOBAL DEFAULT TO float32
torch.set_default_dtype(torch.float32)

def tensor_connection_weight_matrix_R(phase_difference):
    # FORCE phase_difference to float32
    phase_difference = phase_difference.to(dtype=torch.float32)
    
    num_legs = 4  # Assumes there are 4 legs
    R = torch.zeros(8, 8, dtype=torch.float32)

    # Tensorize the computation of rotation matrices - FORCED float32
    phase_diff_matrix = (phase_difference.unsqueeze(0) - phase_difference.unsqueeze(1)).to(dtype=torch.float32)  # [4, 4]
    cos_qji = torch.cos(phase_diff_matrix).to(dtype=torch.float32)  # [4, 4]
    sin_qji = torch.sin(phase_diff_matrix).to(dtype=torch.float32)  # [4, 4]

    for j in range(num_legs):  # rows (legs)
        for i in range(num_legs):  # cols (legs)
            R_block = torch.tensor([
                [cos_qji[j, i], -sin_qji[j, i]],
                [sin_qji[j, i], cos_qji[j, i]]
            ], dtype=torch.float32)  # EXPLICIT float32
            R[2 * j:2 * j + 2, 2 * i:2 * i + 2] = R_block.to(dtype=torch.float32)

    R = torch.round(R, decimals=2).to(dtype=torch.float32)
    return R


@dataclass
class tensorgaitParams:
    H: torch.Tensor           # clearance (mm)
    x_COMshift: torch.Tensor  # shifting for rear legs in x direction (mm)
    robotheight: torch.Tensor # lift off the ground
    dutycycle: torch.Tensor   # duration of stance per gait cycle (0.5-1)
    forwardvel: torch.Tensor  # forward velocity of the bot in mm/s
    T: torch.Tensor           # period of gait cycle in seconds
    yaw_rate: torch.Tensor    # yaw rate in rad/s, useful to make the robot turn
    
    def __post_init__(self):
        """NUCLEAR OPTION: FORCE ALL TENSORS TO float32 - NO EXCEPTIONS!"""
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.H = self.H.to(dtype=torch.float32).float().to(device=self.device)
        self.x_COMshift = self.x_COMshift.to(dtype=torch.float32).float().to(device=self.device)
        self.robotheight = self.robotheight.to(dtype=torch.float32).float().to(device=self.device)
        self.dutycycle = self.dutycycle.to(dtype=torch.float32).float().to(device=self.device)
        self.forwardvel = self.forwardvel.to(dtype=torch.float32).float().to(device=self.device)
        self.T = self.T.to(dtype=torch.float32).float().to(device=self.device)
        self.yaw_rate = self.yaw_rate.to(dtype=torch.float32).float().to(device=self.device)


class VectorizedHopfOscillator:
    def __init__(self, gait_pattern: tensorgaitParams):
        self.gait_pattern = gait_pattern

    def tensor_hopf_cpg_dot(self, Q, R, delta, b, mu, alpha, gamma, dt):
        # NUCLEAR OPTION: FORCE EVERYTHING TO float32 AT THE START
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Q = Q.to(device=device).to(dtype=torch.float32).float()
        R = R.to(device=device).to(dtype=torch.float32).float()        
        # Convert ALL scalars to float32 tensors - NO MERCY
        delta = torch.tensor(float(delta), dtype=torch.float32, device=device)
        b = torch.tensor(float(b), dtype=torch.float32, device=device)
        mu = torch.tensor(float(mu), dtype=torch.float32, device=device)
        alpha = torch.tensor(float(alpha), dtype=torch.float32, device=device)
        gamma = torch.tensor(float(gamma), dtype=torch.float32, device=device)
        dt = torch.tensor(float(dt), dtype=torch.float32, device=device)
        
        num_envs, num_legs_times2 = Q.shape
        num_legs = num_legs_times2 // 2

        x_all = Q[:, 0::2].float()  # [num_envs, num_legs] - FORCED float32
        z_all = Q[:, 1::2].float()  # [num_envs, num_legs] - FORCED float32
        max_val = 100.0

        # Compute scaling factors (<=1)
        x_scale = torch.minimum(torch.ones_like(x_all), max_val / torch.abs(x_all))
        z_scale = torch.minimum(torch.ones_like(z_all), max_val / torch.abs(z_all))

        # Apply scaling
        x_all_clamped = x_all * x_scale
        z_all_clamped = z_all * z_scale

        # Write back to Q
        Q[:, 0::2] = x_all_clamped
        Q[:, 1::2] = z_all_clamped

        # Now compute r^2 safely
        r2 = (x_all_clamped**2 + z_all_clamped**2).float()

        # Broadcast gait params to [num_envs, num_legs] - FORCED float32
        dutycycle = self.gait_pattern.dutycycle.float().unsqueeze(1)  # [num_envs,1]
        T = self.gait_pattern.T.float().unsqueeze(1)

        # FORCED float32 exponentials
        exp_neg = torch.exp(-b * z_all).float()
        exp_pos = torch.exp(b * z_all).float()
        
        stance_denom = (dutycycle * T * (exp_neg + torch.tensor(1.0, dtype=torch.float32, device=device))).float()
        swing_denom = ((torch.tensor(1.0, dtype=torch.float32, device=device) - dutycycle) * T * (exp_pos + torch.tensor(1.0, dtype=torch.float32, device=device))).float()
        
        # FORCED float32 pi
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float32, device=Q.device)
        omega = (pi_tensor / stance_denom + pi_tensor / swing_denom).float()  # [num_envs, num_legs]

        # First term - ALL FORCED float32
        A1 = (alpha * (mu - r2)).float()
        A2 = (gamma * (mu - r2)).float()
        
        # EXPLICIT float32 A matrix
        A = torch.zeros(num_envs, num_legs, 2, 2, dtype=torch.float32, device=Q.device)
        A[:, :, 0, 0] = A1.float()
        A[:, :, 1, 1] = A2.float()
        A[:, :, 0, 1] = (-omega).float()
        A[:, :, 1, 0] = omega.float()

        # FORCED float32 q vector
        q = torch.stack([x_all.float(), z_all.float()], dim=-1).unsqueeze(-1).float()
        
        # THE CRITICAL MATMUL - EVERYTHING IS DEFINITELY float32
        q_dot_first_term = torch.matmul(A.float(), q.float()).squeeze(-1).float()
        
        # FORCED float32 q_dot
        q_dot = torch.zeros_like(Q, dtype=torch.float32).float()
        q_dot[:, ::2] = q_dot_first_term[:, :, 0].float()
        q_dot[:, 1::2] = q_dot_first_term[:, :, 1].float()

        # Second term (coupling) - FORCED float32
        coupling_term = (delta * torch.matmul(R.float(), Q.float().T).T).float()
        q_dot += coupling_term

        # Final result - FORCED float32
        Q_new = (Q.float() + q_dot.float() * dt.float()).float()
        return Q_new


class VectorizedMotionPlanning:
    def __init__(self, gait_pattern: tensorgaitParams, JointOffsets: dict, L1: float, L2: float, z_rest_foot: float):
        self.gait_pattern = gait_pattern
        self.L1 = torch.tensor(float(L1), dtype=torch.float32)  # FORCED float32
        self.L2 = torch.tensor(float(L2), dtype=torch.float32)  # FORCED float32
        self.z_rest_foot = torch.tensor(float(z_rest_foot), dtype=torch.float32)  # FORCED float32
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FORCED float32 joint offsets
        self.x_hipoffset = torch.tensor([JointOffsets[l]["x_offset"] for l in JointOffsets], dtype=torch.float32).float().to(self.device)
        self.z_hipoffset = torch.tensor([JointOffsets[l]["z_offset"] for l in JointOffsets], dtype=torch.float32).float().to(self.device)
        self.y_hipoffset = torch.tensor([JointOffsets[l]["y_offset"] for l in JointOffsets], dtype=torch.float32).float().to(self.device)
        self.isRear = torch.tensor(["Back" in l for l in JointOffsets], dtype=torch.bool).to(self.device)

    def tensor_TrajectoryGenerator(self, x_hopf, z_hopf):
        # FORCE INPUTS TO float32
        x_hopf = x_hopf.to(dtype=torch.float32).float().to(self.device)
        z_hopf = z_hopf.to(dtype=torch.float32).float().to(self.device)
        
        num_envs, num_legs = x_hopf.shape

        # Phase - FORCED float32
        phase_rad = torch.atan2(z_hopf, x_hopf).float()
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float32, device=self.device)
        phase_norm = ((phase_rad + pi_tensor) / (2 * pi_tensor)).float()

        # Broadcast gait params - ALL FORCED float32
        forwardvel = self.gait_pattern.forwardvel.float().unsqueeze(1)
        T = self.gait_pattern.T.float().unsqueeze(1)
        yaw_rate = self.gait_pattern.yaw_rate.float().unsqueeze(1)
        H = self.gait_pattern.H.float().unsqueeze(1)
        dutycycle = self.gait_pattern.dutycycle.float().unsqueeze(1)
        robotheight = self.gait_pattern.robotheight.float().unsqueeze(1)
        x_COMshift = self.gait_pattern.x_COMshift.float().unsqueeze(1)

        # FORCED float32 calculations
        S_body = (forwardvel * T).float()
        dS = (yaw_rate * self.y_hipoffset.float().unsqueeze(0) * T).float()
        S = (S_body + dS).float()
        
        two_pi = torch.tensor(2.0 * torch.pi, dtype=torch.float32, device=self.device)
        x = (S/torch.tensor(2.0, dtype=torch.float32, device=self.device) * torch.cos(two_pi*phase_norm) + self.x_hipoffset.float().unsqueeze(0)).float()
        x = (x + (self.isRear.float().unsqueeze(0) * x_COMshift)).float()

        # Z trajectory - FORCED float32
        half_tensor = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        one_tensor = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        shifted_phase = ((phase_norm + half_tensor) % one_tensor).float()
        swing_mask = (shifted_phase < (one_tensor - dutycycle)).float()

        z = (H * torch.sin(two_pi * shifted_phase) * swing_mask).float()
        z_corrected = (z - H + self.z_rest_foot.float() - robotheight).float()

        return x, z_corrected

    def tensor_InverseKinematics(self, x_array, z_array):
        # FORCE INPUTS TO float32
        x_array = x_array.to(dtype=torch.float32).float()
        z_array = z_array.to(dtype=torch.float32).float()
        
        x_local = (x_array - self.x_hipoffset.float().unsqueeze(0)).float()
        z_local = (z_array - self.z_hipoffset.float().unsqueeze(0)).float()

        r = torch.sqrt(x_local**2 + z_local**2).float()
        
        # FORCED float32 constants
        epsilon = torch.tensor(1e-6, dtype=torch.float32, device=self.device)
        r_min = (torch.abs(self.L1.float() - self.L2.float()) + epsilon).float()
        r_max = (self.L1.float() + self.L2.float() - epsilon).float()
        r = torch.clamp(r, min=r_min, max=r_max).float()

        # FORCED float32 calculations
        two_tensor = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        p = ((self.L2.float()**2 - self.L1.float()**2 - r**2) / (two_tensor*self.L1.float()*r)).float()
        
        neg_one = torch.tensor(-1.0, dtype=torch.float32, device=self.device)
        pos_one = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        p = torch.clamp(p, min=neg_one, max=pos_one).float()

        theta_1 = (torch.arcsin(p) - torch.atan2(z_local, x_local)).float()
        theta_2 = (torch.atan2(-(z_local + self.L1.float()*torch.cos(theta_1)),
                              x_local + self.L1.float()*torch.sin(theta_1)) - theta_1).float()

        return theta_1, theta_2