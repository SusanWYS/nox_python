import numpy as np
import torch
import pandas as pd
import pickle
import scipy
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import read_edf
from preprocess_helpers import plot_spect,channel_select,plot_data,find_data_chn
from model import get_data,gen_nfold_subjs,GRU
from training_helper import butter_low,butter_all,load_data,bandpower
from multitaper_spectrogram_python import multitaper_spectrogram



old_subjs_F4 = ['175','158','167','165','122','107','157','150','122','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4
all_subjects = ['175','158','167','165','122','107','157','150','122','146','104','135','143','152','154','156','177','132','131','171','174','159',"124","128","110","114","119","120","123",'161']
train_losses = np.load("train_losses.npy")
test_losses = np.load("test_rmses.npy")
missing_sub = ['141', '144', '127', '108', '145']
X_data, Y_data, subj_len = get_data(all_subjs_F4)

device = torch.device('cpu')

def create_model(i):
    model = GRU().to(device)
    path = f"/om2/user/susanw26/model_ckpts/subj{all_subjs_F4[i]}.pt"
    def load_ckpt(path,model):
        checkpoint = torch.load(path,map_location = "cpu")
        model.load_state_dict(checkpoint['net_state_dict'])
        return model 
    return load_ckpt(path,model)


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

    #plot_impedance(eog_imp_n, eog_imp,"EOG")
    #plot_impedance(emg_imp_n, emg_imp,"EMG")
    plot_impedance(eeg_imp_n, eeg_imp,"EEG")


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

def plot_all_angles():
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    angs = []
    for i in range(11,len(subj_len)):
        train_x,train_y,val_x,val_y = gen_nfold_subjs(i,subj_len,X_data,Y_data)
        val_x = val_x.detach().numpy()
        x_dat_mean = np.mean(val_x,axis = 1,keepdims = True)
        x_dat_std = np.std(val_x,axis = 1,keepdims = True)
        val_x = torch.tensor(val_x)
        model = create_model(i)
        pred_y = model(val_x).detach().numpy()
        val_y = val_y.detach().numpy()

        Y0 = val_y.reshape((-1,2))
        y_ = pred_y.reshape((-1,2))

        Y0_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in Y0])
        pred_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in y_])
        angs.append(Y0_angles - pred_angles)
    angs = np.concatenate(angs,axis = 0)
    circular_hist(ax,angs)
    return angs,fig, ax

def plot_div_angles(young = True):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize = (30,20))
    all_diffs = np.asarray([])
    start = 0 if young else 12
    end = 12 if young else 19
    for i in range(start,end):
        train_x,train_y,val_x,val_y = gen_nfold_subjs(i,subj_len,X_data,Y_data)
        model = create_model(i)
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
    return 

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
    model = create_model(idx)
    eeg = pd.read_csv(f"ag{all_subjs_F4[idx]}/nox/eeg.csv")

    fig,ax = plt.subplots(2,figsize = (20,8))
    subj_start = np.sum(subj_len[:idx])
    subj_end = np.sum(subj_len[:idx+1])
    X_data = X[subj_start:subj_end]
    Y_data = Y[subj_start:subj_end]
    eeg_chns = np.asarray(eeg.columns)
    F4_idx =np.where(eeg_chns=="F4")[0]
    X0 = butter_all(eeg.values[:,F4_idx].reshape((-1,1)).flatten(),True)
    X0 = X0[start:start+duration*200*5]
    filtered_X0 = butter_low(X0,True)
    train_x,train_y,val_x,val_y = gen_nfold_subjs(0,subj_len,X_data,Y_data)
    pred_y = model(val_x).detach().numpy()
    val_y = val_y.detach().numpy()
    Y0 = val_y[start//1000:start//1000+duration].reshape((-1,2))
    Y0_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in Y0])
    y_ = pred_y[start//1000:start//1000+duration].reshape((-1,2))
    pred_angles = np.asarray([np.angle(i[0]+i[1]*1j) for i in y_])
    x_ticks = np.arange(0,duration*5,0.005)
    raw_x = load_data(all_subjs_F4[idx],"F4")
    opt = dict(color='r',width=5)
    ax[0].plot(x_ticks,X0*1e5,label = "raw",linewidth = 1)
    ax[0].plot(x_ticks,filtered_X0*1e5,label = "filtered",linewidth = 1)
    ax[0].set_ylabel("Potential (uV)", fontsize = 18)
    ax[0].legend(fontsize=18)
    ax[0].set_title("Raw vs Filtered EEG",fontsize=18)
    #ax[1].plot(x_ticks,pred_angles,color = "salmon",label = "Predicted")
    ax[1].plot(x_ticks,Y0_angles,color = "blue")
    ax[1].set_xlabel("Time (s)", fontsize = 18)
    ax[1].set_ylabel("Angle (radian)", fontsize = 18)
    #
    ax[1].set_title("Phase Angles",fontsize=18)
    ax[1].hlines(0,0,30, linestyles=':',color = "r",label = "Peak")
    ax[1].legend(loc='upper right',fontsize=18)
    fig.tight_layout()
    return fig,ax

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

def plot_delta_band(subj):
    subj_eeg = pd.read_csv(f"ag{subj}/nox/eeg.csv")
    subj_f4 = np.array(subj_eeg.loc[:, 'F4'])
    subj_delta = []
    for i in range(0,subj_f4.shape[0],6000):
        data = subj_f4[i:i+6000]
        subj_delta.append(bandpower(data,200,0.1,6,6000))
    subj_delta = np.asarray(subj_delta)
    fig,ax = plt.subplots(2)
    ax[0].plot(subj_delta)
    ax[0].set_xlabel("Sample #")
    ax[0].set_ylabel("Power")
    ax[0].set_title(f"Delta Power for Subj {subj}")
    ax[0].margins(0)
    ax[0].set_ylim(0,3e-9)
    #ax[0].axhline(y=1e-10)
    multitaper_spectrogram(subj_f4,fs = 200,ax = ax[1])
    ax[1].set_ylim(0,20)
    fig.savefig(f"plots/ag{subj}/theta_delta.png")

def plot_stim_window(data):
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
    plt.vlines(200,plt.ylim()[0],plt.ylim()[1],color = 'r',linestyles="--")
    plt.xlabel('Time (s) in relation to simulated stim')
    plt.ylabel('EEG Potential (µV)')
    plt.title('2-s Window Mean with 95% CI')
    # Display the plot
    plt.legend()
    plt.show()

def visualize_sw(duration,start,idx,ax):
    # duration are calculated per 5s. e.g. 1 means 5s, 2 means 10s
    # start are the index of the start segments. Also calculated in 5s
    eeg = pd.read_csv(f"ag{all_subjs_F4[idx]}/nox/eeg.csv")
    eeg_chns = np.asarray(eeg.columns)
    F4_idx =np.where(eeg_chns=="F4")[0]
    X_data = eeg.values
    X0 = butter_all(X_data[start:start+duration*200*5,F4_idx].reshape((-1,1)).flatten(),True)
    filtered_X0 = butter_low(X0,True)

    x_ticks = np.arange(0,duration*5,0.005)
    ax.plot(x_ticks,X0*1e6,label = "Raw")
    ax.plot(x_ticks,filtered_X0*1e6,color = "salmon",label = "Filtered")
    ax.set_xlabel("Time (s)",fontsize = 18)
    ax.set_ylabel("Voltage (µV)",fontsize = 18)
    ax.legend(fontsize = 18)
    ax.set_title("Slow Wave EEG",fontsize = 18)

def plot_sim_angles(stim_angles,subj_num):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    circular_hist(ax,stim_angles)
    stim_num = len(stim_angles)
    plt.figtext(0.5, -0.05, f"Total Num of Stims Per Subj: {stim_num/subj_num}", ha="center", fontsize=18)
    plt.tight_layout()
    return fig,ax

# for subj in missing_sub:
#     plot_subj_sleep(subj)
#     plot_imp(subj)

# for subj in all_subjs_F4:
#     plot_delta_band(subj)

#plot_all_angles()

#plot_all_loss()
#y = visualize_model_output(X_data,Y_data,6,100,14)
#visualize_data(X_data, Y_data,6,3000,14)
#visualize_model_output(X_data, Y_data,6,100,1)
#plot_div_angles()

# fig,ax = plt.subplots(figsize = (20,8))
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Angle (radian)")
# visualize_sw(X_data, Y_data,6,12000*200//5,12,ax,"young","blue")

#fig,ax = visualize_model_output(X_data, Y_data,6,8530*200,0)
# fig,ax = plt.subplots(figsize = (15,6))
# visualize_sw(6,8530*200,0,ax)
# fig.tight_layout