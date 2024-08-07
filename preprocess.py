import read_edf
from preprocess_helpers import channel_select, plot_data, trim, save, rereference, plot_spect,find_sleep_interval,find_data_chn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
import pandas as pd

new_young_subjs = []
new_old_subjs = []

old_subjs_d = ['175','158',"167",'165','122','107','157','150','122','146',"104","108","127","135","141","143","144","145",'152'] #Done
young_subjs_d = ['154','156','177','132','131','171','174','159',"124","128","110","114","119","120","123",'161']
def preprocess_data(subj):
    # load subject data
    eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
    print(f"Data imported for {subj}.")

    # select eeg, eog, emg data
    eeg_chns, eeg_sig, eeg_imp_n, eeg_imp = channel_select(eeg,ratio_thresh=0.05)
    eog_chns, eog_sig, eog_imp_n, eog_imp = channel_select(eog,ratio_thresh=0.3)
    emg_chns, emg_sig, emg_imp_n, emg_imp = channel_select(emg,ratio_thresh=0.3)
    with open(f"ag{subj}/nox/good channels", "w") as fp:
        json.dump(list(eeg_chns), fp)
    print("Channels selected.")

    # rereference eeg, emg, and eog
    eeg_reref = rereference(eeg_chns, eeg_sig)
    eog_reref = rereference(eog_chns, eog_sig)
    emg_reref = rereference(emg_chns, emg_sig)
    print("Channels rereferenced.")

    # choose interval to trim data based on sleep and wake-up time
    data_to_plot,chns_to_plot = find_data_chn(misc,eog_imp,emg_imp,chan_names)
    start,end  = find_sleep_interval(data_to_plot,chns_to_plot)

    # save the start and end time:
    with open(f"ag{subj}/nox/sleep interval", "w") as fp:
        json.dump([start,end], fp)
    
    # trim data
    eeg_reref_trim = trim(eeg_reref,start,end)
    eog_reref_trim = trim(eog_reref,start,end)
    emg_reref_trim = trim(emg_reref,start,end)
    physio_vals = np.asarray(list(physio.values()))
    physio_chns = np.asarray(list(physio.keys()))
    physio_vals = trim(physio_vals,start,end)

    # save eeg + eog + emg for sleep scoring 
    save(eeg_chns,eeg_reref_trim,"eeg",subj)
    save(emg_chns,emg_reref_trim,"emg",subj)
    save(eog_chns,eog_reref_trim,"eog",subj)
    save(physio_chns,physio_vals,"physio",subj)
    print(f"data saved for {subj}")


def plot_subj_data(subj,chn_name):
    # Reread edf and find imp and misc
    eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
    eog_chns, eog_sig, eog_imp_n, eog_imp = channel_select(eog,ratio_thresh=0.3)
    emg_chns, emg_sig, emg_imp_n, emg_imp = channel_select(emg,ratio_thresh=0.3)
    data_to_plot,chns_to_plot = find_data_chn(misc,eog_imp,emg_imp,chan_names)
    # Read Preprocessed Data
    eeg = pd.read_csv(f"ag{subj}/nox/eeg.csv")
    eeg_chns = np.asarray(eeg.columns)
    eeg_sig = eeg.values.T
    # extract start and end time
    start, end = json.load(open(f"ag{subj}/nox/sleep interval", "r"))
    # find if the selected chan is in the good channels
    chn_exist = chn_name in eeg_chns
    if chn_exist:
        chn_idx = np.where(eeg_chns == chn_name)[0][0]
        fig,ax = plt.subplots(1 + len(chns_to_plot),figsize = (8,4*(1 + len(chns_to_plot))))
        # plot imp and misc
        plot_data(data_to_plot,chns_to_plot,x_min_max = (start,end),ax = ax[:-1])
        # plot spectro
        plot_spect(subj,[chn_name], eeg_sig[chn_idx],axis = ax[-1])
        fig.suptitle(f"subj {subj} chn {chn_name}")
        fig.tight_layout()
        fig.savefig(f"ag{subj}/nox/{chn_name}.png")
        print(f"spect plotted for {subj}")
    chn_name = "F4"
    chn_exist = chn_name in eeg_chns
    if chn_exist:
        chn_idx = np.where(eeg_chns == chn_name)[0][0]
        fig,ax = plt.subplots(1 + len(chns_to_plot),figsize = (8,4*(1 + len(chns_to_plot))))
        # plot imp and misc
        plot_data(data_to_plot,chns_to_plot,x_min_max = (start,end),ax = ax[:-1])
        # plot spectro
        plot_spect(subj,[chn_name], eeg_sig[chn_idx],axis = ax[-1])
        fig.suptitle(f"subj {subj} chn {chn_name}")
        fig.tight_layout()
        fig.savefig(f"ag{subj}/nox/{chn_name}.png")
        print(f"spect plotted for {subj}")

for subj in old_subjs_d:
    plot_subj_data(subj,"F3")

for subj in young_subjs_d:
    plot_subj_data(subj,"F3")