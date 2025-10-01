from scipy.signal import butter,filtfilt

def lowpass(data,cutoff,sample_rate,order=4):
    nyq=0.5*sample_rate #Nyquist frequency
    normalcutoff=cutoff/nyq
    b,a=butter(order,normalcutoff,btype='low',analog=False)
    filtered_signal=filtfilt(b,a,data)
    return filtered_signal
