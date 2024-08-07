import sys
import os 
import numpy as np
import mne 
import pandas as pd

def get_nox_data_from_edf(subject):
    if subject == '128' or subject == '114':
        fname =  f"ag{subject}/nox/edf/full_night.edf"
    elif subject == '124':
        fname =  f"ag{subject}/nox/fullnight.edf"
    elif subject == "104":
        fname =  f"ag{subject}/nox/ag104.edf"
    elif subject == "152":
        fname = f"ag{subject}/nox/full_night_ag152.edf"
    elif subject == "161":
        fname = f"ag{subject}/nox/ag161_full_night.edf"
    else:
        fname =  f"ag{subject}/nox/full_night.edf"

    # Load raw EDF
    raw = mne.io.read_raw_edf(fname, preload=True)
    chan_names = raw.ch_names
    raw_data = raw.get_data()

    # Get EEG data
    eeg_chans = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'M1', 'M2']
    impedance_channels = []
    for i, name in enumerate(eeg_chans):
        impedance_channels.append(eeg_chans[i] + ' Impedance')
        
    eeg_dict = {}
    for i, name in enumerate(eeg_chans+impedance_channels):
        if name not in chan_names:
            continue
        eeg_dict[name] = raw_data[chan_names.index(name),:]
        
    # Get EOG data
    eog_dict = {}
    for i, name in enumerate(['E1', 'E2', 'E1 Impedance', 'E2 Impedance']):
        if name not in chan_names:
            continue
        eog_dict[name] = raw_data[chan_names.index(name),:]
        
    # Get EMG data
    emg_dict = {}
    for i, name in enumerate(['1', '2', 'F', '1 Impedance', '2 Impedance', 'F Impedance']):
        if name not in chan_names:
            continue
        emg_dict[name] = raw_data[chan_names.index(name),:]

    # Get physio data
    physio_dict = {}
    physio_signals = ['ECG', 'cRIP Flow', 'Thorax', 'Abdomen', 'Flow', 'Saturation', 'Pulse', 'Pulse Waveform']
    for signal in physio_signals:
        if signal in chan_names:
            physio_dict[signal] = raw_data[chan_names.index(signal),:]
        else:
            print(f'No {signal} signal in NOX data')
                
    # Get misc signals
    misc_dict = {}
    misc_signals = ['Activity', 'Light', 'Snore', 'PosAngle', 'Left Leg', 'Right Leg', 'Left Leg Impedan', 'Right Leg Impeda']

    for signal in misc_signals:
        if signal in chan_names:
            misc_dict[signal] = raw_data[chan_names.index(signal),:]
        else:
            print('No ' + signal + ' signal in NOX data')


    return eeg_dict, eog_dict, emg_dict, physio_dict, misc_dict, raw_data, chan_names
