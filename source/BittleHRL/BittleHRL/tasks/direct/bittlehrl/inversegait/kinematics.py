import numpy as np

# robot measurements obtained using Digimizer in mm
hiplength=47.9 #(L1)
kneelength= 46.5 #(L2)
bodywidth= 90.7
bodylength= 104
hipz=-21.02  ## to be tweaked after

# dictionary data type, defining the limbs relative to the center of the robot
JointOffsets={"Right Front":{"x_offset":bodylength/2, "y_offset":-bodywidth/2, "z_offset":hipz,"Hip Index":9, "Knee Index":13}, 
              "Left Front":{"x_offset":bodylength/2, "y_offset":bodywidth/2, "z_offset":hipz,"Hip Index":8, "Knee Index":12},
              "Right Back":{"x_offset":-bodylength/2, "y_offset":-bodywidth/2, "z_offset":hipz,"Hip Index":10, "Knee Index":14},
              "Left Back":{"x_offset":-bodylength/2, "y_offset":bodywidth/2, "z_offset":hipz,"Hip Index":11, "Knee Index":15}}

# x0,y0,z0= offset from origin to hip joint
# theta 1= angle of hip joint (radians)
# L1= hip length
# L2= knee length
# theta 2= angle of knee relative to hip (radians)

# find the foot position relative to the robot origin even as it moves

def HomogeneousTransforms(Legname,runarray,L1=hiplength,L2=kneelength):
    if type(Legname)!= str:
        print('AHHHH IT HURTS')
    else:
        x0=JointOffsets[Legname]["x_offset"]
        y0=JointOffsets[Legname]["y_offset"]
        z0=JointOffsets[Legname]["z_offset"]
        placements=[JointOffsets[Legname]["Hip Index"]-8, JointOffsets[Legname]['Knee Index']-8]
        theta1= runarray[placements[0]]
        theta2=runarray[placements[1]]

        T_offset=np.array([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
        Rot1=np.array([[np.cos(theta1),0,np.sin(theta1),0],[0,1,0,0],[-np.sin(theta1),0,np.cos(theta1),0],[0,0,0,1]])
        T_hiptoknee=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-L1],[0,0,0,1]])
        Rot2=np.array([[np.cos(theta2),0,np.sin(theta2),0],[0,1,0,0],[-np.sin(theta2),0,np.cos(theta2),0],[0,0,0,1]])
        T_kneetofoot=np.array([[1,0,0,L2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    return T_offset@Rot1@T_hiptoknee@Rot2@T_kneetofoot
