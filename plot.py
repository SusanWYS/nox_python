import numpy as np
import torch
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import read_edf
from preprocess_helpers import plot_spect,channel_select,plot_data,find_data_chn
from model import get_data,gen_nfold_subjs,GRU
from prep_training import circular_hist,butter_low,load_data
from multitaper_spectrogram_python import multitaper_spectrogram

old_subjs_F4 = ['175','158','167','165','122','107','157','150','122','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4
all_subjects = ['175','158','167','165','122','107','157','150','122','146','104','135','143','152','154','156','177','132','131','171','174','159',"124","128","110","114","119","120","123",'161']
train_losses = np.load("train_losses.npy")
test_losses = np.load("test_rmses.npy")

X_data, Y_data, subj_len = get_data(all_subjs_F4)

def plot_imp(subj):
    eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
    eog_chns, eog_sig, eog_imp_n, eog_imp = channel_select(eog,ratio_thresh=0.3)
    emg_chns, emg_sig, emg_imp_n, emg_imp = channel_select(emg,ratio_thresh=0.3)
    eeg_chns, eeg_sig, eeg_imp_n, eeg_imp = channel_select(eeg,ratio_thresh=0.05)
    
    def plot_impedance(chn_n,imp,name = None):
        num = imp.shape[0]
        if num == 0:
            return
        fig,ax = plt.subplots(num,squeeze = False)
        for i in range(num):
            ax[i][0].plot(imp[i],label = chn_n[i])
            ax[i][0].legend(loc="upper right")
            ax[i][0].set_ylabel("Ω")
            ax[i][0].set_xlabel("Sample")
        fig.suptitle(f"Subj {subj} {name} Impedance")
        fig.tight_layout()
        fig.savefig(f"plots/ag{subj}/{name} Impedance.png")

    plot_impedance(eog_imp_n, eog_imp,"EOG")
    plot_impedance(emg_imp_n, emg_imp,"EMG")
    plot_impedance(eeg_imp_n, eeg_imp,"EEG")


def load_ckpt(path,model,multi = False):
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    if not multi:
        model.load_state_dict(checkpoint['net_state_dict'])
    else:
        ckpts = OrderedDict()
        for key in checkpoint['net_state_dict']:
            ckpts[key[7:]] = checkpoint['net_state_dict'][key]
        model.load_state_dict(ckpts)
    return model

def plot_sing_loss(subj,x1,x2,ax):
    ax.plot(x1,label = "Train")
    ax.plot(x2,label = "Test")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss")
    ax.set_title(f"Subject {subj} Loss")

def plot_all_loss(save = True):
    fig, ax = plt.subplots(5,4,figsize=(11.69,8.27))
    for i in range(len(train_losses)):
        subj = all_subjs_F4[i]
        x1 = train_losses[i]
        x2 = test_losses[i]
        j = i%5
        k = i%4
        plot_sing_loss(subj,x1,x2,ax[j][k])
    labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(labels_handles.values(),labels_handles.keys())
    fig.tight_layout()
    if save:
        fig.savefig("All Losses.png")


def plot_all_angles(save = True):
    fig, ax = plt.subplots(5,4, subplot_kw=dict(projection='polar'),figsize = (30,20))
    for i in range(19):
        j = i % 5
        k = i % 4
        train_x,train_y,val_x,val_y = gen_nfold_subjs(i,subj_len,X_data,Y_data)
        val_x = val_x.detach().numpy()
        x_dat_mean = np.mean(val_x,axis = 1,keepdims = True)
        x_dat_std = np.std(val_x,axis = 1,keepdims = True)
        val_x = torch.tensor((val_x - x_dat_mean)/x_dat_std)
        model = GRU()
        model = load_ckpt(f"model_ckpts/sub{all_subjs_F4[i]}.pkl",model)
        pred_y = model(val_x).detach().numpy()
        val_y = val_y.detach().numpy()

        Y0 = val_y.reshape((-1,2))
        y_ = pred_y.reshape((-1,2))

        Y0_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in Y0])
        pred_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in y_])
        ang_diff = Y0_angles - pred_angles
        circular_hist(ax[j][k],ang_diff)
        ax[j][k].set_title(f"Subj {all_subjs_F4[i]}")
    if save:
        fig.savefig("plots/ang_err.png")

def plot_div_angles(young = True):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize = (30,20))
    all_diffs = np.asarray([])
    start = 0 if young else 12
    end = 12 if young else 19
    for i in range(start,end):
        train_x,train_y,val_x,val_y = gen_nfold_subjs(i,subj_len,X_data,Y_data)
        model = GRU()
        model = load_ckpt(f"model_ckpts/sub{all_subjs_F4[i]}.pkl",model)
        pred_y = model(val_x).detach().numpy()
        val_y = val_y.detach().numpy()

        Y0 = val_y.reshape((-1,2))
        y_ = pred_y.reshape((-1,2))

        Y0_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in Y0])
        pred_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in y_])
        ang_diff = Y0_angles - pred_angles
        all_diffs = np.hstack((ang_diff,all_diffs))
    circular_hist(ax,all_diffs)
    if young:
        ax.set_title(f"Young Subject")
    else:
        ax.set_title(f"Old Subject")



def visualize_data(X,Y,duration,start,idx):
    # duration are calculated per 5s. e.g. 1 means 5s, 2 means 10s
    # start are the index of the start segments. Also calculated in 5s
    fig,ax = plt.subplots(3,figsize = (20,12))
    subj_start = np.sum(subj_len[:idx])
    subj_end = np.sum(subj_len[:idx+1])
    X_data = X[subj_start:subj_end]
    Y_data = Y[subj_start:subj_end]
    X0 = X_data[start//1000:start//1000+duration,:,0].reshape((-1,1)) 
    filtered_X0 = butter_low(X0.flatten(),True)

    Y0 = Y_data[start//1000:start//1000+duration].reshape((-1,2))
    Y0_angles = [np.angle(i[0]+i[1]*1j) for i in Y0]
    x_ticks = np.arange(0,duration*5,0.005)
    raw_x = load_data(all_subjs_F4[idx],"F4")
    multitaper_spectrogram(raw_x,fs = 200,ax = ax[0])
    opt = dict(color='r',width=5)
    ax[0].annotate('',xy=(start,20),xycoords='data',xytext =(start,30),textcoords = 'data',arrowprops=opt)
    ax[0].set_title("Spectrogram") 
    ax[0].set_ylim(0,30)
    ax[1].plot(x_ticks,X0,label = "raw",linewidth = 1)
    ax[1].plot(x_ticks,filtered_X0,label = "filtered",linewidth = 1)
    ax[1].set_ylabel("Potential (uV)")
    ax[1].set_title("Raw vs Filtered Data")
    ax[2].set_title("Phase Angle")
    ax[2].plot(x_ticks,Y0_angles)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (radian)")

    fig.legend()
    fig.tight_layout()

def visualize_model_output(X,Y,duration,start,idx):
    # duration are calculated per 5s. e.g. 1 means 5s, 2 means 10s
    # start are the index of the start segments. Also calculated in 5s
    model = GRU()
    model = load_ckpt(f"model_ckpts/sub{all_subjs_F4[idx]}.pkl",model)
    
    fig,ax = plt.subplots(3,figsize = (20,12))
    subj_start = np.sum(subj_len[:idx])
    subj_end = np.sum(subj_len[:idx+1])
    X_data = X[subj_start:subj_end]
    Y_data = Y[subj_start:subj_end]
    X0 = X_data[start//1000:start//1000+duration,:,0].reshape((-1,1)) 
    filtered_X0 = butter_low(X0.flatten(),True)
    train_x,train_y,val_x,val_y = gen_nfold_subjs(0,subj_len,X_data,Y_data)
    pred_y = model(val_x).detach().numpy()
    val_y = val_y.detach().numpy()
    Y0 = val_y[start//1000:start//1000+duration].reshape((-1,2))
    y_ = pred_y[start//1000:start//1000+duration].reshape((-1,2))
    Y0_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in Y0])
    pred_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in y_])
    x_ticks = np.arange(0,duration*5,0.005)
    raw_x = load_data(all_subjs_F4[idx],"F4")
    multitaper_spectrogram(raw_x,fs = 200,ax = ax[0])
    opt = dict(color='r',width=5)
    ax[0].annotate('',xy=(start,20),xycoords='data',xytext =(start,30),textcoords = 'data',arrowprops=opt)
    ax[0].set_title("Spectrogram") 
    ax[0].set_ylim(0,30)
    ax[1].plot(x_ticks,X0,label = "raw",linewidth = 1)
    ax[1].plot(x_ticks,filtered_X0,label = "filtered",linewidth = 1)
    ax[1].set_ylabel("Potential (uV)")
    ax[2].plot(x_ticks,pred_angles,color = "salmon",label = "Predicted")
    ax[2].plot(x_ticks,Y0_angles,color = "blue",label = "Target")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle (radian)")
    fig.legend()
    fig.tight_layout()

def plot_subj_eeg_data(subj,chn_name):

    # Read Preprocessed Data
    eeg = pd.read_csv(f"ag{subj}/nox/eeg.csv")
    eeg_chns = np.asarray(eeg.columns)
    eeg_sig = eeg.values.T
    # extract start and end time
    start, end = 0, eeg.shape[0]
    # find if the selected chan is in the good channels
    chn_exist = chn_name in eeg_chns
    if chn_exist:
        chn_idx = np.where(eeg_chns == chn_name)[0][0]
        fig,ax = plt.subplots()
        # plot spectro
        plot_spect(subj,[chn_name], eeg_sig[chn_idx],axis = ax)
        fig.suptitle(f"subj {subj} chn {chn_name}")
        fig.savefig(f"plots/ag{subj}/{chn_name}.png")
        print(f"spect plotted for {subj}")

def plot_subj_sleep(subj):
    # Reread edf and find imp and misc
    eeg, eog, emg, physio, misc, raw_data, chan_names = read_edf.get_nox_data_from_edf(subj)
    eog_chns, eog_sig, eog_imp_n, eog_imp = channel_select(eog,ratio_thresh=0.3)
    emg_chns, emg_sig, emg_imp_n, emg_imp = channel_select(emg,ratio_thresh=0.3)
    data_to_plot,chns_to_plot = find_data_chn(misc,eog_imp,emg_imp,chan_names)

    # extract start and end time
    start, end = json.load(open(f"ag{subj}/nox/sleep interval", "r"))

    # plot imp and misc
    fig,ax = plt.subplots(len(chns_to_plot),figsize = (8,4*(1 + len(chns_to_plot))))
    plot_data(data_to_plot,chns_to_plot,x_min_max = (start,end),ax = ax)
    fig.tight_layout()
    fig.savefig(f"plots/ag{subj}/sleep.png")
    print(f"fig plotted for {subj}")


for subj in all_subjects :
    plot_subj_sleep(subj)


#plot_all_angles()
#plot_all_loss()
# visualize_model_output(X_data,Y_data,6,100,0)
#visualize_data(X_data, Y_data,6,3000,14)
#visualize_model_output(X_data, Y_data,6,8200,0)
#plot_div_angles()