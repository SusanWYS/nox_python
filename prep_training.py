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
        # x_dat_mean = np.mean(x_dat,axis = 2,keepdims = True)
        # x_dat_std = np.std(x_dat,axis = 2,keepdims = True)
        # x_dat = (x_dat - x_dat_mean)/x_dat_std
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

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches






#visualize_data(X_data,Y_data,6,100)

#----------
# model = load_ckpt(Path("model_ckpts/subj175.pkl"),GRU())
# train_x,train_y,val_x,val_y = gen_nfold_subjs(0,subj_len,X_data,Y_data)
# pred_y = model(train_x).detach().numpy()
# val_y = train_y.detach().numpy()
# Y0 = val_y.reshape((-1,2))
# y_ = pred_y.reshape((-1,2))
# Y0_angles = [np.angle(i[0]+i[1]*1j) for i in Y0]
# pred_angles = [np.angle(i[0]+i[1]*1j) for i in y_]

# fig,ax = plt.subplots(2,figsize = (20,16))
# xticks = np.linspace(0,5,1000)
# ax[0].plot(xticks,pred_y[0])
# ax[0].set_title("Predicted")
# ax[1].plot(xticks,val_y[0])
# ax[1].set_title("Target")

# fig,ax = plt.subplots(2,figsize = (20,16))
# xticks = np.linspace(0,5,1000)
# ax[0].plot(xticks,pred_angles[:1000])
# ax[0].set_title("Predicted")
# ax[1].plot(xticks,Y0_angles[:1000])
# ax[1].set_title("Target")

