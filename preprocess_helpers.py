# %% Imports and necessary variables
import numpy as np
import matplotlib.pyplot as plt 
import read_edf
from multitaper_spectrogram_python import multitaper_spectrogram
import pandas as pd
import os
from os.path import dirname, abspath
from multitaper_spectrogram_python import multitaper_spectrogram
from pathlib import Path

# %% Helper Functions

def plot_data(data,chns,x_min_max = (0,0),ax = None):
    '''
    Plot the data based on the given args.
    Args:
        data (act,light,emg_imp_avg,eog_imp_avg): the data to plot
        chns (1d array): the names of channels to plot
        x_min_max (tuple): (xmin, xmax) of a horizontal line
        ax (False or matplotlib axes): if provided, plot onto the axes
    Return:
        ax or (fig,ax) 
    '''
    # extract start and end time (min and max)
    x_min = x_min_max[0]
    x_max = x_min_max[1]
    # if axes are given
    if ax is not None:
        for i in range(data.shape[0]): 
            ax[i].plot(data[i],label = chns[i])
            ax[i].hlines(y = 0.005,xmin = x_min, xmax = x_max, color = 'red', lw = 3)
            ax[i].set_xlabel('Sample #')
            ax[i].legend(loc="upper right")
        return ax
    # if no axes are given, we create a new plot
    fig, ax = plt.subplots(data.shape[0],squeeze=False)
    for i in range(data.shape[0]): 
        ax[i,0].plot(data[i],label = chns[i])
        ax[i,0].hlines(y = 0.5,xmin = x_min, xmax = x_max, color = 'red', lw = 3)
        ax[i,0].set_xlabel('Sample #')
        ax[i,0].legend(loc="upper right")
    return (fig,ax)

def channel_select(data, threshold = 50000,ratio_thresh = 0.15):
    """
    Select good channels to use based on impedance level.
    Args:
        data (dict): data to select channels
    Return:
        good_chns (list): selected channel names
        good_dat (2d array): selected signal data
        good_imp_names (list): selected channel impedance names
        good_imp (2d array): selected impedance data
    """
    channel_names = np.asarray(list(data.keys())[:len(data.keys())//2])
    channel_dat = np.asarray([data[chn] for chn in channel_names]) 
    impedance_names = np.asarray(list(data.keys())[len(data.keys())//2:])
    impedance_dat = np.asarray([data[chn] for chn in impedance_names])

    # calculate how many times each channel impedance has gown over the threshold
    overshoot_num = np.sum(impedance_dat > threshold,axis = 1)
    overshoot_ratio = overshoot_num/impedance_dat.shape[1]

    good_chns = channel_names[np.where(overshoot_ratio < ratio_thresh)[0]]
    good_dat = channel_dat[np.where(overshoot_ratio < ratio_thresh)[0]]
    good_imp_names = impedance_names[np.where(overshoot_ratio < ratio_thresh)[0]]
    good_imp = impedance_dat[np.where(overshoot_ratio < ratio_thresh)[0]]

    return good_chns, good_dat, good_imp_names, good_imp


def rereference(chn_names,data):
    """
    Rereference the data
    Args:
        chn_names (list or 1d array): names of good channels
        data (2d array): data to rereference
    Return: 
        reref_sig (2d array): rereferenced signal
    """
    mastoids = ["M1" in chn_names,"M2" in chn_names]
    if all(mastoids): # If both mastoids channel are good
        baseline_sig = np.mean(data[-2:],axis= 0)
    elif any(mastoids): # If only one mastoid channel is good
        baseline_sig = data[-1:]
    else: # Else, re-reference to the average of good channels
        baseline_sig = np.mean(data,axis=0)
    reref_sig = data - baseline_sig
    return reref_sig


def trim(data,start,end):
    """
    Trim the data
    Args:
        data (1d or 2d array): data to trim
        start (int): the starting sample num of sleep 
        end (int): the ending sample num of sleep
    Return: the trimmed data (same shape as the input data)
    """
    if data.ndim == 2: # if it's a 2d array
        return data[:,start:end]
    return data[start:end] # if it's a 1d array


def plot_spect(subj,chns, sig, fs = 200,frequency_range = [0, 30],time_bandwidth = 15,
               num_tapers = 29,window_params = [30,  5],min_nfft = 0,
               detrend_opt = 'constant',multiprocess = True,n_jobs = 3,
               weighting = 'unity',plot_on = True,return_fig = False,
               clim_scale = False,verbose = True,xyflip = False,axis = None):
    # if no axis is given, we create and return a new plot
    if axis is None:
        figure,axes = plt.subplots(len(chns),figsize = (8,4*len(chns)),squeeze = False)
        for idx,chn in enumerate(chns):
            multitaper_spectrogram(sig[idx], fs, frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, n_jobs,
                                                    weighting, plot_on, return_fig, clim_scale, verbose, xyflip,ax = axes[idx,0])
            axes[idx,0].set_title(f"channel {chn}")
        figure.suptitle(f"Subject {subj}")
        figure.tight_layout()
        return figure,axes
    # if axis is given, we plot on the gien axis (return nothing)
    else:
        multitaper_spectrogram(sig, fs, frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, n_jobs,
                                                    weighting, plot_on, return_fig, clim_scale, verbose, xyflip,ax = axis)
        axis.set_title(f"channel {chns[0]}")



def save(data_keys,data_vals,dat_name,subj):
    """
    Save data by a given file name to a specific path
    Args:
        data_keys (list or 1d array): the channels
        data_vals (2d array): the data values
        dat_name (str): the type of the data to store (eeg/eog/emg/physio)
        subj (str): the subject number
    Return none
    """
    target_path = Path(os.path.join(os.path.abspath("/rdma/vast-rdma/vast/lewislab/susanw26"),
                               f"ag{subj}/nox"))
    data_dict = {} # create a new dictionary to store data
    for i in range(len(data_keys)):
        data_dict[data_keys[i]] = data_vals[i,:]
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(os.path.join(target_path,f"{dat_name}.csv"),index=False)

def find_data_chn(misc,eog_imp,emg_imp,chan_names):
    data_to_plot = np.zeros((4,eog_imp.shape[1]))
    if "Activity" in chan_names:
        data_to_plot[0] = misc["Activity"]
    if "Light" in chan_names:
        data_to_plot[1] = misc["Light"]
    data_to_plot[2] = np.mean(eog_imp,axis = 0)
    data_to_plot[3] = np.mean(emg_imp,axis = 0)
    chns_to_plot = np.asarray(["Activity","Light","Eog Imp","Emg Imp"])

    non_zero_ind = []
    for i in range(4):
        if all(data_to_plot[i] == np.zeros(eog_imp.shape[1])):
            continue
        non_zero_ind.append(i)

    return data_to_plot[non_zero_ind],chns_to_plot[non_zero_ind]

def find_sleep_interval(data_to_plot,chns_to_plot):
    response = None
    plot_data(data_to_plot,chns_to_plot,"Sleep Interval")
    while response != "y":
         # prompt the user to enter sleep start and end time based on visualiztion
         start, end = list(map(int,input("Manually enter start and end time of sleep:").split()))
         plot_data(data_to_plot,chns_to_plot,"Sleep Interval",(start,end))
         response = input("Sleep interval looks  good? (enter \"y\" if good)")
    return start,end
