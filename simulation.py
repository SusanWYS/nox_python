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
                

import read_edf
from preprocess_helpers import channel_select, plot_data, trim, save, rereference, plot_spect,find_sleep_interval,find_data_chn
from training_helper import butter_all,circular_hist,butter_low,bandpower
from multitaper_spectrogram_python import multitaper_spectrogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import time
import random
import gc
import json
import torch
import scipy
import pandas as pd
import scipy
import torch.nn as nn


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
old_subjs_F4 = ['175','158','167','165','107','157','150','122','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4
test_young_subjs_F4 = ['159', '114']
test_old_subjs_F4 = ['143','152']

# need to train a new model based on all the validation subjects and test them on the test subjects
class GRU(nn.Module):

    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(3,16)
        self.gru2 = nn.GRU(16,4)
        self.fc = nn.Linear(4, 2)


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

    def forward(self, x, h1, h2):
        x,h1 = self.gru1(x, h1)
        x,h2 = self.gru2(x)
        x = self.fc(x)
        return x,h1,h2



class simulator():

    def __init__(self,subj):
        self.subj = subj
        self.stim_window = None
        self.eeg, self.act, self.light = self.input_data(subj)
        self.eeg = butter_all(self.eeg,True)
        self.hilb_angles = np.asarray([np.angle(i) for i in scipy.signal.hilbert(self.eeg)])
        self.stim_idx = 0
        self.stim_idces = np.asarray([],dtype=int)
        self.h1 = torch.zeros(1, 1000, 16)
        self.h2 = torch.zeros(1, 1000, 4)
        

    def input_data(self,subj):
        eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
        data = [eeg["F4"],None,None]

        if "Activity" in misc.keys():
            data[1] = misc["Activity"]
        else:
            data[1] = np.zeros_like(eeg["F4"])
        if "Light" in misc.keys():
            data[2] = misc["Light"]
        else:
            data[2] = np.zeros_like(eeg["F4"])
        return data

    def create_model(self):
        model = GRU()
        if self.subj in set(['143','152','159', '114']):
            path = f"model_ckpts/subj154.pkl"
        else:
            path = f"model_ckpts/subj{self.subj}.pkl"
        def load_ckpt(path,model):
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            model.load_state_dict(checkpoint['net_state_dict'])
            return model 
        return load_ckpt(path,model)
 
    def check_asleep(self,act,light):
        if any(act > 0.4) or any(light > 1):
            return False
        return True

    def check_stimulate_pt(self,data,new_stim_idx):
        if new_stim_idx > self.stim_idx + 200 * 2: # check
            conf = np.linalg.norm(data[0]+1j*data[1]) # look at the vector that I generated (if I thresholded, if I'm only looking at those with norm larger than 0.8. Play around the threshold. )
            ang = np.abs(np.angle(data[0]+1j*data[1]))
            if conf > 0.7:
                #print(ang)
                if 0.2 * np.pi > ang or ang > (np.pi + 0.8 * np.pi):
                    return True

    def generate_data(self):
        idx = 0
        has_data = True
        while has_data:
            chunk_num = 400 # random.randint(5,10)
            yield chunk_num
            idx += chunk_num
            has_data = idx < self.eeg.shape[0]
    
    def simulate(self):
        data_generator = self.generate_data()
        idx = 0
        model = self.create_model().to(device)

        for chunk_num in data_generator:
            idx += chunk_num 
            delt_pass = False
            if idx > 6000: 
                act = self.act[idx-chunk_num:idx]
                light = self.light[idx-chunk_num:idx]
                eeg = self.eeg[idx-1000:idx]
                if self.check_asleep(act,light):
                    delt_power = bandpower(self.eeg[idx-6000:idx], 200, 0.1, 6,6000)
                    delt_pass = delt_power > 8e-10
                    # measure delta power in the past 30 seconds
                    if not delt_pass:
                        continue
                    # create inputs
                    filtered_eeg = butter_low(eeg)
                    eeg_diff = np.hstack((np.diff(eeg),np.array(eeg[-1]-eeg[-2])))
                    inputs = np.vstack((eeg,filtered_eeg,eeg_diff))
                    inputs_mean = np.mean(inputs,axis = 1,keepdims = True)
                    inputs_std = np.std(inputs,axis = 1,keepdims = True)
                    inputs = (inputs - inputs_mean)/inputs_std
                    inputs = np.expand_dims(inputs.T, axis=0)[:,-chunk_num:]
                    inputs = torch.from_numpy(inputs).to(device)
                    
                    # feed the inputs into the model
                    h1 = self.h1[:,-chunk_num:,:].to(device)
                    h2 = self.h2[:,-chunk_num:,:].to(device)
                    outputs,h1,h2 = model(inputs,h1,h2)
                    outputs = outputs.cpu().detach().numpy()
                    h1 = h1.cpu().detach()
                    h2 = h2.cpu().detach()
                    self.h1 = torch.cat((self.h1,h1),1)
                    self.h2 = torch.cat((self.h2,h2),1)

                    if self.check_stimulate_pt(outputs[0,-1],idx):
                        self.stim_idces = np.append(self.stim_idces,idx)
                        self.stim_idx = idx
                    del inputs
                    del h1
                    del h2
                    del outputs

            gc.collect()
            torch.cuda.empty_cache()

        
    def plot_sim_angles(self,save = True):
        stim_angles = self.hilb_angles[self.stim_idces]
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        circular_hist(ax,stim_angles)
        stim_num = len(self.stim_idces)
        plt.figtext(0.5, -0.05, f"Total Num of Stims: {stim_num}", ha="center", fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(f"plots/ag{self.subj}/stim_angles.png")

    def plot_stim_spect(self, save = True):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        n = self.eeg.shape[0]
        ax2.set_xlim(0,n)
        ax2.set_xticks(np.arange(0, n + 1, 720000))
        ax2.set_xticklabels([int(tick / 720000) for tick in np.arange(0, n + 1, 720000)])
        ax2.set_xlabel('Time (Hrs)')
        multitaper_spectrogram(self.eeg,fs = 200,ax = ax1)
        ax1.set_ylim(0,25)
        ax1.set_xticks(np.arange(0, n//200 + 1, 3600))
        ax1.set_xticklabels([int(tick / 3600) for tick in np.arange(0, n//200 + 1, 3600)])
        ax1.set_xlabel('Time (Hrs)')
        plt.suptitle("Sleep Spectrogram with Stimulation Points")
        for i in self.stim_idces:
            ax2.axvline(x = i, color = 'r')
        if save:
            plt.savefig(f"plots/ag{self.subj}/stim_spects.png")

    def find_stim_win(self):
        stim_window = []
        for i in self.stim_idces:
            stim_window.append(self.eeg[i-200:i+200])
        self.stim_window = stim_window
    
    def plot_stim_window(self,save = True):
        self.find_stim_win()
        data = np.asarray(self.stim_window)  # m rows, n columns
        m,n = data.shape
        # Calculate the mean and SEM for each column
        means = np.mean(data, axis=0)*1e6
        sems = scipy.stats.sem(data, axis=0)*1e6
        # Calculate the 95% confidence interval
        confidence_interval = 1.96 * sems  # 95% CI
        # Plot the means with error bars representing the 95% confidence interval
        x = np.arange(n)
        plt.plot(x,means)
        plt.xticks([0, 200, 400], [-1, 0, 1])
        plt.fill_between(x, (means-confidence_interval), (means+confidence_interval), color='b', alpha=.1)
        # Add labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('EEG Potential (µV)')
        plt.title('2-s Window Mean with 95% CI')
        # Display the plot
        plt.legend()
        plt.show()
        if save:
            plt.savefig(f"plots/ag{self.subj}/stim_window.png")
                
old_train_angs = []
young_train_angs = []

old_test_angs = []
young_test_angs = []

young_train_stim_wind = []
old_train_stim_wind = []

young_test_stim_wind = []
old_test_stim_wind = []



############## simulate old training sujbects #############

# for i in old_subjs_F4:
#     sim = simulator(i)
#     sim.simulate()
#     sim.plot_stim_window()
#     sim.plot_sim_angles()
#     sim.plot_stim_spect()
#     if sim.stim_idces.size == 0:
#         continue
#     stim_angles = sim.hilb_angles[sim.stim_idces]
#     old_train_angs += stim_angles.tolist()
#     old_train_stim_wind.append(sim.stim_window)

# old_train_stim_wind = np.concatenate(old_train_stim_wind,axis = 0)
# np.save('old_train_stim_wind.npy', old_train_stim_wind)
# np.save("old_train_angs.npy",np.asarray(old_train_angs))

# for i in young_subjs_F4:
#     sim = simulator(i)
#     sim.simulate()
#     sim.plot_stim_window()
#     sim.plot_sim_angles()
#     sim.plot_stim_spect()
#     if sim.stim_idces.size == 0:
#         continue
#     stim_angles = sim.hilb_angles[sim.stim_idces]
#     young_train_angs += stim_angles.tolist()
#     young_train_stim_wind.append(sim.stim_window)

# young_train_stim_wind = np.concatenate(young_train_stim_wind,axis = 0)
# np.save('young_train_stim_wind.npy', young_train_stim_wind)
# np.save("young_train_angs.npy",np.asarray(young_train_angs))



############### simulate old test subjects ##############

# for i in test_old_subjs_F4:
#     sim = simulator(i)
#     sim.simulate()
#     sim.plot_stim_window()
#     sim.plot_sim_angles()
#     sim.plot_stim_spect()
#     if sim.stim_idces.size == 0:
#         continue
#     stim_angles = sim.hilb_angles[sim.stim_idces]
#     old_test_angs  += stim_angles.tolist()
#     old_test_stim_wind.append(sim.stim_window)

# old_test_stim_wind = np.concatenate(old_test_stim_wind,axis = 0)
# np.save('old_test_stim_wind.npy', old_test_stim_wind)
# np.save("old_test_angles.npy",np.asarray(old_test_angs))



############### simulate young training subjects #############

# sim154 = simulator(154)
# a = time.time()
# sim154.simulate()
# b = time.time()
# print((b-a)/60)
# sim154.plot_stim_window()
# sim154.plot_sim_angles()
# sim154.plot_stim_spect()

# for subj in young_subjs_F4:
#     sim = simulator(subj)
#     fig,ax = plt.subplots(2)
#     ax[0].plot(sim.act)
#     ax[1].plot(sim.light)
#     ax[0].set_ylim(0,0.4)
#     ax[1].set_ylim(0,1)
#     fig.show()


############ simulate young test subjects ##############

# for i in test_young_subjs_F4:
#     sim = simulator(i)
#     sim.simulate()
#     sim.plot_stim_window()
#     sim.plot_sim_angles()
#     sim.plot_stim_spect()
#     if sim.stim_idces.size == 0:
#         continue
#     stim_angles = sim.hilb_angles[sim.stim_idces]
#     young_test_angs += stim_angles.tolist()
#     young_test_stim_wind.append(sim.stim_window)

# young_test_stim_wind = np.concatenate(young_test_stim_wind,axis = 0)
# np.save('young_test_stim_wind.npy', young_test_stim_wind)
# np.save("young_test_angles.npy",np.asarray(young_test_angs))


####### PLOTING ############

# def plot_sim_angles(stim_angles,subj_num):
#     fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#     circular_hist(ax,stim_angles)
#     stim_num = len(stim_angles)
#     plt.figtext(0.5, -0.05, f"Total Num of Stims Per Subj: {stim_num/subj_num}", ha="center", fontsize=12)
#     plt.tight_layout()
#     return fig,ax

# fig1,ax1 = plot_sim_angles(np.load("young_train_angs.npy"),7)
# ax1.set_title("Phase Angle Err for Young Train Subjs")
# fig2,ax2 = plot_sim_angles(np.load("old_train_angs.npy"),11)
# ax2.set_title("Phase Angle Err for Old Train Subjs")

# def plot_stim_window(data):
#     m,n = data.shape
#     # Calculate the mean and SEM for each column
#     means = np.mean(data, axis=0)*1e6
#     sems = scipy.stats.sem(data, axis=0)*1e6
#     # Calculate the 95% confidence interval
#     confidence_interval = 1.96 * sems  # 95% CI
#     # Plot the means with error bars representing the 95% confidence interval
#     x = np.arange(n)
#     plt.plot(x,means)
#     plt.xticks([0, 200, 400], [-1, 0, 1])
#     plt.fill_between(x, (means-confidence_interval), (means+confidence_interval), color='b', alpha=.1)
#     # Add labels and title
#     plt.vlines(200,plt.ylim()[0],plt.ylim()[1],color = 'r',linestyles="--")
#     plt.xlabel('Time (s) in relation to simulated stim')
#     plt.ylabel('EEG Potential (µV)')
#     plt.title('2-s Window Mean with 95% CI')
#     # Display the plot
#     plt.legend()
#     plt.show()

# plot_stim_window(np.load("old_test_stim_wind.npy"))