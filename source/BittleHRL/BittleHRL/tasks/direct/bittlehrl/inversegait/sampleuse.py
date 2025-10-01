import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

run1=pd.read_csv('angle_log_trotForwardrun1.csv')

from inversegait import preprocessing
processangles_radians= preprocessing.RawJointAngleProcessed(run1) #angles preprocessed to radians

from inversegait import frequencyanal
gaitF=np.mean(frequencyanal.gaitfrequency(run1,'trot 1')) # getting the gair frequency in Hz

from inversegait import foot_positions
footpos_mm,time=foot_positions.LegSeparationFootPositions(run1)

from inversegait import stanceDC
stancedutycycles,pp,_,_,_=stanceDC.duty_cycle_compute(run1) # getting the duty cycles for the data

from inversegait import StrideLength
strideL=StrideLength.stridelength(footpos_mm)

from inversegait import swingheight
swingH,_=swingheight.clearanceheight(footpos_mm,time,pp)
print(swingH)

from inversegait import PD
phased_legs=PD.phasedifference(footpos_mm,time,run1)
