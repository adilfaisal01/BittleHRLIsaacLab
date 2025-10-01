from scipy.signal import find_peaks
import numpy as np

def stridelength(FTT):
    avg_SL_per_foot=[]
    for i in range(4):
        stride_lengths=[]
        xpos=np.array(FTT[i,:,0]) # extracting x coordinates for foot position, first coordinate dictates what foot is used
        xpos_centered=xpos-np.mean(xpos)

        # finding the peaks and troughs of the signal
        peakindices,_=find_peaks(xpos_centered)
        troughindices,_=find_peaks(-xpos_centered)

        peaks=xpos_centered[peakindices]
        troughs=xpos_centered[troughindices]
    

        for j in range(min(len(peaks),len(troughs))):
            SL=peaks[j]-troughs[j]
            stride_lengths.append(SL)

        avg_SL=np.mean(stride_lengths)
        avg_SL_per_foot.append(avg_SL)
    return avg_SL_per_foot

# print(f'Average Stride Length of each foot in mm: {np.round(avg_SL_per_foot,3)}')

