# building a data set of foot positions for each foot, looping all the feet on the quadruped utilising the dictionary
import matplotlib.pyplot as plt
import numpy as np
from .preprocessing import RawJointAngleProcessed
from .kinematics import JointOffsets, HomogeneousTransforms

def LegSeparationFootPositions(run2):

    time=run2['timestamp'] # system time
    time=list(time)
    xxs=RawJointAngleProcessed(run2) # fully processed data for foot position calculations
    t_total = xxs.shape[1]  # number of time steps
    FTT=[]

    for name in JointOffsets.keys():
        foot_positions = []
        for t in range(t_total):
            pose=xxs[:,t]
            T=HomogeneousTransforms(name,runarray=pose)
            foot_positions.append(T[:,3])

        foot_positions_xyz=np.array(foot_positions)[:,0:3]  # final foot positions in space
        FTT.append(foot_positions_xyz)


        # plt.plot(list(time), foot_positions_xyz[:, 0],label='x',color='r') 
        # plt.plot(list(time), foot_positions_xyz[:, 1],label='y',color='b') 
        # plt.plot(list(time), foot_positions_xyz[:, 2],label='z',color='k') 

        # plt.xlabel('time (s)')
        # plt.ylabel('position (mm)')
        # plt.title(f'{name} Foot Trajectory Relative to robot origin')
        # plt.grid(True)
        # plt.legend()
        # plt.show()
    FTT=np.stack(FTT,axis=0)

    return FTT,time # slapping all the foot positions for each position into a single 3d array with shape (4,timesteps,3), 4 for the number of feet, 3 for the xyz axes respectively
