import math
import numpy as np

def Quarternion2EulerAngles(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    roll_x=2*(w*x+y*z)
    roll_y=1-2*(x**2+y**2)
    roll=math.atan2(roll_x,roll_y)
    sinp = 2 * (w * y - z * x)
    if sinp > 1:
        sinp = 1
    elif sinp < -1:
        sinp = -1
    pitch = math.asin(sinp)
    yawx=2*(w*z+y*x)
    yawy=1-2*(y**2+z**2)
    yaw=math.atan2(yawx,yawy)
    return np.array([roll,pitch,yaw])

