# based off zhenget al's work on modified Hopf Oscillators for CPG and then sine based trajectory and Inverse Kinematics of the 
import numpy as np
from dataclasses import dataclass


def connectionwieghtmatrixR(phase_difference):
    R=np.zeros((8,8)) # is=ts 4x4 matrix with each block being 2x2 so the total dimensions end up being 8x8
    for j in range(4): #rows
        for i in range(4): #columns
            qji=phase_difference[i]-phase_difference[j]
            Rji=np.array([[np.cos(qji),-np.sin(qji)],[np.sin(qji),np.cos(qji)]])
            R[2*j:2*j+2,2*i:2*i+2]=Rji

    R=np.round(R,decimals=2)
    return R




## leg geometric parameters included in the MotionPlanning class
## the trjaectories are generated for each foot individually-- modify 

# insert data class for gait_Params and RobotGeometry
# phase difference can be taken care of by the rotation matrix in the hopf cpg function


@dataclass
class gaitParams:
    H: float #clearance (mm)
    x_COMshift:float #shifting for rear legs in x direction (mm)
    robotheight: float #lift off the ground
    dutycycle:float #duration of stance per gait cycle (0.5-1)
    forwardvel:float #forward velocity of the bot in mm/s
    T: float #period of gait cycle in seconds
    yaw_rate:float #yaw rate in rad/s, useful to make the robot turn


class HopfOscillator:
    def __init__(self,gait_pattern:gaitParams):
        self.gait_pattern=gait_pattern
    
    def hopf_cpg_dot(self,Q,R,delta,b,mu,alpha,gamma,dt):
        q_dot=np.zeros(8) #Q=[x1,y1,x2,y2,x3,y3,x4,y4]

        for i in range(4):
            xi=Q[2*i]
            zi=Q[2*i+1]
            q=np.array([[xi],[zi]])
            r2=xi**2+zi**2
            stance_denom=self.gait_pattern.dutycycle*self.gait_pattern.T*(np.exp(-b*zi)+1)
            swing_denom=(1-self.gait_pattern.dutycycle)*self.gait_pattern.T*(np.exp(b*zi)+1)
            omega=np.pi/stance_denom+np.pi/swing_denom
            A=np.array([[alpha*(mu-r2),-omega],[omega,gamma*(mu-r2)]])
            q_dot_first_term=A@q  # covers the first term in equation (9) from Zeng et. al
            q_dot[2*i:2*i+2]=q_dot_first_term.flatten()
        
        # second term
        q_dot += delta * R @ Q  
        Q_new=Q+q_dot*dt
        return Q_new

class MotionPlanning:
    def __init__(self,gait_pattern:gaitParams,x_hipoffset,z_hipoffset,isRear,L1,L2,z_rest_foot,y_hipoffset):
        self.gait_pattern=gait_pattern 
        self.x_hipoffset=x_hipoffset #resting position of the foot
        self.z_rest_foot=z_rest_foot  
        self.z_hipoffset=z_hipoffset #derived from JointOffsets dictionary
        self.y_hipoffset=y_hipoffset #derived from the Jointoffsets dictionary
        self.L1=L1 #shoulder (hip) length in mm
        self.L2=L2 #elbow (knee) length in mm
        self.isRear=isRear #is it a rear leg
        self.globalx=0 #global position of the x coord for the foot
        

    def TrajectoryGenerator(self,x_hopf,z_hopf):
        XX=[]
        ZZ=[]
        S_body=self.gait_pattern.forwardvel*self.gait_pattern.T
        dS=self.gait_pattern.yaw_rate*self.y_hipoffset*self.gait_pattern.T
        S=S_body+dS
        for i in range(len(x_hopf)):
            phase_rad=np.arctan2(z_hopf[i],x_hopf[i]) #hopf oscillator tells the phase of the leg in radians 
            phase_norm=(phase_rad+np.pi)/(2*np.pi)
            if self.isRear:
                x=S/2*np.cos(2*np.pi*phase_norm)+self.x_hipoffset+self.gait_pattern.x_COMshift
            else:
                x=S/2*np.cos(2*np.pi*phase_norm)+self.x_hipoffset #S= stride length (mm)

            shifted_phase_norm=(phase_norm+0.5) %1 ## renormalize the phase norm since the original phase normalized has a half a cycle worth of discrepancy
            if shifted_phase_norm < (1-self.gait_pattern.dutycycle):
                z = self.gait_pattern.H * np.sin(2 * np.pi * shifted_phase_norm) #H is swing height (mm), swing phase runs this part
                
            else:
                z = 0  # stance phase

            z_corrected=z-self.gait_pattern.H+self.z_rest_foot-self.gait_pattern.robotheight #corrected to be absolute position of the foot relative to robot origin
            XX.append(x)
            ZZ.append(z_corrected)
            

        return XX,ZZ
    
    # updating the global position of the robot's feet, also provides a framework for adaptation to terrain by verying the z velocity being introduced as a parameter
    def globalFootPos(self,x_relarr,z_relarr,dt):
        x_global=[]
        z_global=[]
        
        for x_rel,z_rel in zip(x_relarr,z_relarr):
            self.globalx+=self.gait_pattern.forwardvel*dt
            x_global.append(x_rel + self.globalx)
            z_global.append(z_rel)
        return x_global,z_global

# x_abs=S/2*np.cos(2*np.pi*phase_norm)+x_hipoffset+x_COMshift #S= stride length (mm)
#x_COMshift is mainly to indicate the shift seen during gait movement since during analysis it was noticed that the rear hips moved further from their initial position


    def InverseKinematics(self,x_array,z_array):
        theta1_list=[] #list of hip angles (radians)
        theta2_list=[] #list of knee angles (radians)
        for x,z in zip(x_array,z_array):
            # removing the hip offsets
            x_local=x-self.x_hipoffset
            z_local=z-self.z_hipoffset
            
            # L1= hip to knee length, L2= knee to foot length
            r=np.sqrt(x_local**2+z_local**2)
            r = np.clip(r, abs(self.L1 - self.L2) + 1e-6, self.L1 + self.L2 - 1e-6)  # small epsilon to avoid singularities
            p=(self.L2**2-self.L1**2-r**2)/(2*self.L1*r)
            p=np.clip(p,-1.0,1.0)
            theta_1=np.arcsin(p)-np.arctan2(z_local,x_local)
            theta_2=np.arctan2(-(z_local+self.L1*np.cos(theta_1)),x_local+self.L1*np.sin(theta_1))-theta_1

            theta1_list.append(theta_1)
            theta2_list.append(theta_2)

        return theta1_list,theta2_list