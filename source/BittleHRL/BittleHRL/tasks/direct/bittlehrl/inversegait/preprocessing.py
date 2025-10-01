# servo angles are given in degrees, so this function converts to radians and does all the processing for the joints from the bittle


import numpy as np
from .kinematics import JointOffsets

def RawJointAngleProcessed(run):
    runjoint=[]
    for i in range(8,16):
        angle_radians=np.array(np.deg2rad(run[f'joint_{i}']))
        runjoint.append(angle_radians)
    indices_flipper=[]
    for name,info in JointOffsets.items():
        if "Right" in name:
            print (f"{name} CCW is -ve for you")
            indices=[info["Hip Index"], info["Knee Index"]]
            indices_flipper.append(indices)
                        
    indices_flipper=np.array(indices_flipper).flatten()-8  ## since we are staring the count leg joint, we discarded the first 8 joints from the petoi bittle for now

    for i in range(0,8):
        if i in indices_flipper:
            runjoint[i]=runjoint[i]*1 # correction for the right side so that it follows the CCW positive convention
        else:
            runjoint[i]=runjoint[i]*1 # left side CCW is already +ve, so no need for correction here
    return np.array(runjoint)
