import sympy as sp
import numpy as np

def InverseKinematics(x,z,xoffset,zoffset,L1,L2):
    # removing the offset
    x_local=x-xoffset
    z_local=z-zoffset
    
    # L1= hip to knee length, L2= knee to foot length
    r=np.sqrt(x_local**2+z_local**2)
    p=(L2**2-L1**2-r**2)/(2*L1*r)
    theta_1=np.arcsin(p)-np.arctan2(z_local,x_local)
    theta_2=np.arctan2(-(z_local+L1*np.cos(theta_1)),x_local+L1*np.sin(theta_1))-theta_1

    return theta_1,theta_2


## Derived by solving the FK final equations as shown below through geometric methods (shown in IK_derivation.pdf)
## Symbolic forward Kinematics



# symbol definition
L1,L2=sp.symbols('L1  L2')
theta1,theta2=sp.symbols('theta1 theta2')
x0,y0,z0=sp.symbols('x0 y0 z0')

# Forward Kinematics to find foot positions
T_offset=sp.Matrix([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
Rot1=sp.Matrix([[sp.cos(theta1),0,sp.sin(theta1),0],[0,1,0,0],[-sp.sin(theta1),0,sp.cos(theta1),0],[0,0,0,1]])
T_hiptoknee=sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,-L1],[0,0,0,1]])
Rot2=sp.Matrix([[sp.cos(theta2),0,sp.sin(theta2),0],[0,1,0,0],[-sp.sin(theta2),0,sp.cos(theta2),0],[0,0,0,1]])
T_kneetofoot=sp.Matrix([[1,0,0,L2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

T_total=T_offset@Rot1@T_hiptoknee@Rot2@T_kneetofoot
T_total.simplify()
T_total[:,3] # final x,y,z positions of each foot (including the offset)
