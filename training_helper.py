import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pickle
from multitaper_spectrogram_python import multitaper_spectrogram


def load_data(subj,chn_name):
    eeg = pd.read_csv(f"ag{subj}/nox/eeg.csv")                
    eeg_chns = np.asarray(eeg.columns)
    idx = np.where(eeg_chns == chn_name)[0][0]
    eeg_sig = eeg.values.T
    return eeg_sig[idx]

def butter_low(data,phase0 = False):
    fs = 200
    N,wn = scipy.signal.buttord(wp=(0.4,2), ws = (0.1,4),gpass=0.5, gstop=2.5,fs=fs)
    b, a = scipy.signal.butter(N,wn, btype='bandpass', fs = fs)
    if phase0: 
        return scipy.signal.filtfilt(b, a, data)
    return scipy.signal.lfilter(b, a, data)

def butter_all(data,phase0 = True):
    fs = 200
    N,wn = scipy.signal.buttord(wp=(0.4,20), ws = (0.1,25),gpass=0.5, gstop=2.5,fs=fs)
    b, a = scipy.signal.butter(N,wn, btype='bandpass', fs = fs)
    if phase0: 
        return scipy.signal.filtfilt(b, a, data)
    return scipy.signal.lfilter(b, a, data)

def prep_data(idx_lis,chn_name):
    """
    Prepare preprocessed data for training
    Args:
        idx_lis: a list of participants indices
    """
    subj_data = []
    for idx in idx_lis:
        dat = load_data(idx,chn_name)
        dat = butter_all(dat)
        filtered_dat = butter_low(dat)
        dat_diff = np.hstack((np.diff(dat),np.array(dat[-1]-dat[-2])))
        data = np.vstack((dat,filtered_dat,dat_diff))
        subj_data.append(data)
    X_data = []
    Y_data = []
    subj_len = []
    for data in subj_data:
        filtered_y = butter_low(data[0,20:],True)
        y_dat = scipy.signal.hilbert(filtered_y)
        y_dat = np.asarray([[np.real(y)/np.linalg.norm(y),np.imag(y)/np.linalg.norm(y)] for y in y_dat]).T
        x_dat = segment(data[:,:-20],5)
        #x_dat[:,0,:] = 
        x_dat_mean = np.mean(x_dat,axis = 2,keepdims = True)
        x_dat_std = np.std(x_dat,axis = 2,keepdims = True)
        x_dat = (x_dat - x_dat_mean)/x_dat_std
        y_dat = segment(y_dat,5)
        X_data.append(x_dat)
        Y_data.append(y_dat)
        subj_len.append(y_dat.shape[0])
    X_data = np.concatenate(X_data)
    Y_data = np.concatenate(Y_data)
    return X_data,Y_data,subj_len

def segment(data,length):
    seg_num = data.shape[1]//(200*length)
    trim_data = data[:,:seg_num*(200*length)]
    return np.asarray(np.split(trim_data,seg_num,axis=1))
