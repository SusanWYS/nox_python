import read_edf
from preprocess_helpers import channel_select, plot_data, trim, save, rereference, plot_spect,find_sleep_interval,find_data_chn
from prep_training import butter_all
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import scipy
import pandas as pd
import time
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

old_subjs_F4 = ['175','158','167','165','122','107','157','150','122','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4

# need to train a new model based on all the validation subjects and test them on the test subjects
class GRU(nn.Module):

    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(3,16)
        self.gru2 = nn.GRU(16,4)
        self.fc = nn.Linear(4, 2)
        self.h1 = torch.zeros(1, 1000, 16)
        self.h2 = torch.zeros(1, 1000, 4)

    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.GRU:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        device = x.get_device()
        self.h1 = self.h1.to(device)
        self.h2 = self.h2.to(device)

        x,h1 = self.gru1(x,self.h1)
        self.h1 = h1

        x,h2 = self.gru2(x,self.h2)
        self.h2 = h2

        x = self.fc(x)
        return x



class simulator():
    def __init__(self,subj,frequency = 200,chunk_per_sec = 4):
        self.subj = subj
        self.frequency = frequency
        self.chunk_per_sec = chunk_per_sec
        self.eeg, self.act, self.light = self.input_data(subj)
        self.hilb_eeg = scipy.signal.hilbert(butter_all(self.eeg,True))
        self.hilb_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in self.hilb_eeg])
        self.stim_idx = 0

    def input_data(self,subj):
        eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
        data = (eeg["F4"],None,None)

        if "Activity" in raw_data.keys():
            data[1] = raw_data["Activity"]
        if "Light" in raw_data.keys():
            data[2] = raw_data["Light"]
        return data

    def create_model(self):
        model = GRU()
        path = f"model_ckpts/subj{self.subj}.pkl"
        def load_ckpt(path,model):
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            model.load_state_dict(checkpoint['net_state_dict'])
            return model 
        return load_ckpt(path,model)
 
    def check_asleep(self,act,light):
        if any(act > 2) or any(light > 20):
            return False
        return True

    def check_stimulate_pt(self,data,new_stim_idx):

        if new_stim_idx > self.stim_idx + self.frequency * 2: # check
            conf = np.norm(data[0]+1j*data[1])
            if conf > 1:
                ang = np.angle(data[0]+1j*data[1])
                if 0.1 * np.pi > ang and ang > -0.1 * np.pi:
                    self.stim_idx = new_stim_idx
                    return ang

    def generate_data(self):
        idx = 0
        chunk_num = self.frequency//self.chunk_per_sec
        while True:
            data_point = (self.eeg[idx:idx+chunk_num],self.act[idx:idx+chunk_num],self.light[idx:idx+chunk_num])
            yield data_point
            idx += chunk_num
            
            time.sleep(1 / self.chunk_per_sec)
    
    def simulate(self):
        model = self.create_model().to(device)
        data_generator = self.generate_data()
        idx = 0
        for data in data_generator:
            eeg,act,light = data
            idx += len(eeg)
            eeg = butter_all(eeg)
            eeg = torch.from_numpy(eeg).to(device)
            if self.check_asleep(act,light):
                outputs = model(eeg).detach().numpy()
                ang = self.input_datacheck_stimulate_pt(outputs[-1],idx)
                if ang is not None:
                    sec = idx // self.frequency
                    ang_err = ang - self.hilb_angles[idx]
                    print(f"Stimulate at {sec} s with an angle error of {ang_err} radians")
                

