## Phase difference between legs, cross correlation will be used between foot positions, as a sanity check, servo angles will be also compared
## use the front left as base leg 
## Front Right F 0
# Front Left F 1 
# rear right 2 
# rear left 3

## normalize all the data by removing the DC offset in the dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate,correlation_lags
from .frequencyanal import gaitfrequency


def phasedifference(FTT,time,run2):

    FR=FTT[0,:,0]-np.mean(FTT[0,:,0])
    FL=FTT[1,:,0]-np.mean(FTT[1,:,0])
    RR=FTT[2,:,0]-np.mean(FTT[2,:,0])
    RL=FTT[3,:,0]-np.mean(FTT[3,:,0])

    legs=[FR,FL,RR,RL]
    print(np.shape(legs))

    # plt.plot(time,FL,label='front left')
    # plt.plot(time,RR,label='rear right')
    # plt.plot(time,RL,label='rear left')
    # plt.plot(time,FR,label='front right')
    # plt.legend(bbox_to_anchor=(1.05,1))
    # plt.show()

    Freq=len(FR)/(time[-1]-time[0])

    Phase_D=[]
    for i in range(0,4):
        Comp_leg=legs[i]
        xcorr=correlate(FL,Comp_leg,method='direct') # cross correlation metric
        lags=correlation_lags(len(FL),len(Comp_leg)) # find the lags

        # plt.plot(lags,xcorr)
        # plt.show()

        wheremax=np.max(xcorr)
        lag=lags[xcorr==wheremax]
        # print(lag)


    # conversion to time in seconds
        time_diff_s=lag/Freq
    # now convert it phase difference 
        gaitperiod=1/(np.mean(gaitfrequency(run2,'xxq'))) # gait period
        Phase_difference=(time_diff_s/gaitperiod)%1
        Phase_D.append(Phase_difference)


    return list(Phase_D)
