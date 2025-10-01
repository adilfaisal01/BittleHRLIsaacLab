import numpy as np
from scipy. fft import  fft, fftfreq
from scipy.signal.windows import hann
from scipy.signal import find_peaks

def gaitfrequency(run,runname):
    samplefreq=len(run['timestamp'])/run['timestamp'].iloc[-1]
    FundamentalFrequency=[]
    for i in range(8,16):
        LFH= np.array(run[f'joint_{i}'])
        LFH=(LFH-np.mean(LFH))
        N=len(LFH)
        w=hann(len(LFH))
        lfh_windowed=w*LFH
        lfh_fft=fft(lfh_windowed)
        xf=fftfreq(N,1/samplefreq)

        #Keep only the positive frequencies
        pos_mask = xf >=0
        fft_freqs = xf[pos_mask]
        fft_magnitude = (2.0 / N) * np.abs(lfh_fft[pos_mask])

        # Plot magnitude spectrum
        # plt.figure(figsize=(10, 4))
        # plt.plot(fft_freqs, fft_magnitude)
        # plt.title(f'FFT of Joint {i} {runname}')
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()


        # finding the peak frequencies from the FFT
        peaks_amps=find_peaks(fft_magnitude,height=0.1*max(fft_magnitude))
        peakfreqs=fft_freqs[peaks_amps[0]]
        fundfreq=min(peakfreqs)
        FundamentalFrequency.append(fundfreq)
    return FundamentalFrequency
