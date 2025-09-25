# Clearance Height calculation
import numpy as np
import matplotlib.pyplot as plt
from .filters import lowpass
# Clearance is defined as the largest height the robot foot moves to move forward, can be calculated using the z position of the foot
def clearanceheight(FTT,time,centroids):
    zpos=FTT[0,:,2] # finding the z position for each foot, first coordinate determines the foot number
    stance_centroid=centroids[0][0]
    # unlike the x position data, the z data is noisy for the robot so a filter will be applied before processing is completed to even out the roughness at the peaks
    zpos_mean=np.mean(zpos) # mean
    zpos_std=np.std(zpos) #standard deviation
    zpos_stance=stance_centroid*zpos_std+zpos_mean # ground position in mm, or at least inference when foot hits ground

    zpos_centered=zpos-zpos_stance # removing the bias off the signal
    # steps:
    # 1. apply filter-- Lowpass
    # 2. find peaks and use that information to find the clearance

    # using FFT to detect the frequencies in the signal
    # from scipy.fft import fft, fftfreq
    Freq=len(zpos_centered)/(time[-1]-time[1]) # calculating frequency in hz
    T=1/Freq #period in seconds



    ## apply low pass filter and compare to the OG signal
    z_filtered=lowpass(zpos_centered,cutoff=4.5,sample_rate=Freq)
    # plt.plot(time,z_filtered,label='filtered signal',color='r')
    # plt.plot(time,zpos_centered,label='OG signal',color='b',linestyle='--')
    # plt.grid()
    # plt.xlabel('time (sec)')
    # plt.ylabel('z position (mm)')
    # plt.axhline(0, label='ground foot position from K means clustering',color='g')
    # plt.legend(bbox_to_anchor=(1,1))
    # plt.show()


    ## for walk gait, the cutoff frequency is 1.9 Hz, and for trot is 4.5 Hz
    # for the clearance, find the max height off the ground

    from scipy.signal import find_peaks
    troughs_height,_=find_peaks(-z_filtered,height=0)
    mean_trough=np.abs(np.mean(z_filtered[troughs_height]))

    peaks_z,_=find_peaks(z_filtered,height=0)
    mean_peak=np.mean(z_filtered[peaks_z])

    clearance=mean_peak+mean_trough # clearance (max height off ground in mm during swing)
    return clearance,z_filtered
