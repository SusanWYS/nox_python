from prep_training import prep_data
from collections import OrderedDict
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import torch.utils.data as datas
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path
from ray import init
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle


def get_data(subj_list, save = False):
    if save:
        X_data,Y_data,subj_len = prep_data(subj_list,"F4")
        np.save("X_data", X_data)
        np.save("Y_data", Y_data)
        np.save("subj_len",subj_len)
    else:
        X_data = np.load("X_data.npy")
        Y_data = np.load("Y_data.npy")
        subj_len = np.load("subj_len.npy")

    X_data = np.moveaxis(X_data,1,-1)
    Y_data = np.moveaxis(Y_data,1,-1)
    
    return X_data, Y_data, subj_len

# get train and test data for F4 channel
old_subjs_F4 = ['175','158','167','165','122','107','157','150','122','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4
#tests_F4_old = ['143','152',]
#tests_F4_young = ['159', '114']

X_data, Y_data, subj_len = get_data(all_subjs_F4)
torch.set_default_dtype(torch.float64)

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
        # self.h1 = self.h1.to(device)
        # self.h2 = self.h2.to(device)
        x,h1 = self.gru1(x,self.h1)
        x,h2 = self.gru2(x,self.h2)
        x = self.fc(x)
        return x
 
def gen_nfold_subjs(val_subj_idx,subj_len,X_train,Y_train):
    torch.set_default_dtype(torch.float64)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    idx_total = 0

    for i in range(len(subj_len)):
        if i == val_subj_idx:
            val_x.append(X_train[idx_total:idx_total+subj_len[i]])
            val_y.append(Y_train[idx_total:idx_total+subj_len[i]])
        else:
            train_x.append(X_train[idx_total:idx_total+subj_len[i]])
            train_y.append(Y_train[idx_total:idx_total+subj_len[i]])
        idx_total += subj_len[i]

    train_x = torch.tensor(np.concatenate(train_x))
    train_y = torch.tensor(np.concatenate(train_y))
    val_x = torch.tensor(np.concatenate(val_x))
    val_y = torch.tensor(np.concatenate(val_y))

    return train_x,train_y,val_x,val_y

train_losses = []
test_losses = []
def train_kfold(config,train_x,train_y,test_x,test_y,idx):
    torch.set_default_dtype(torch.float64)
    start_time = time.time()
    print(f"Subject {all_subjs_F4[idx]} training started.")
    net = GRU()
    net.init_weights()
    # Initialize GRU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    test_x,test_y = test_x.to(device), test_y.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    trainset = datas.TensorDataset(train_x,train_y)
    trainloader = datas.DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    
    train_loss = []
    test_loss = []
    for i in range(config["epoch"]):
        net.train()
        for inputs, targets in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        net.eval()
        with torch.no_grad():
            outputs = net(train_x)
            device = outputs.get_device()
            train_y = train_y.to(device)
            train_rmse = criterion(outputs, train_y).item()
            train_loss.append(train_rmse)
            print(f"Epoch {i} Train Loss: {train_rmse}")

            test_y_pred = net(test_x)
            device = test_y_pred.get_device()
            test_y = test_y.to(device)
            test_rmse = criterion(test_y_pred, test_y).item()
            test_loss.append(test_rmse)
            print(f"Epoch {i} Test Loss: {test_rmse}")

    end_time = time.time()
    train_time = (end_time - start_time)/60 #mins
    print(f"Subject {all_subjs_F4[idx]} training finished in {train_time} mins.")
    
    # Save checkpoint
    checkpoint_data = {
        "epoch": config["epoch"],
        "batch_size":config["batch_size"],
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
                {"loss": test_rmse},
                checkpoint=checkpoint,
        )
    
    # Save losses
    train_losses.append(train_loss)
    test_losses.append(test_loss)



def main(heldout_idx, num_samples, gpus_per_trial=1):
    torch.set_default_dtype(torch.float64)
    train_x,train_y,test_x,test_y = gen_nfold_subjs(heldout_idx,subj_len,X_data,Y_data)
    config = {
        "batch_size": tune.choice([64]),
        "epoch": tune.choice([10]),
    }
    result = tune.run(
        tune.with_parameters(train_kfold,train_x=train_x,train_y=train_y,test_x=test_x,test_y=test_y,idx = heldout_idx),
        resources_per_trial={"cpu": 15,"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
         data_path = Path(checkpoint_dir) / "data.pkl"
         with open(data_path, "rb") as fp:
             best_checkpoint_data = pickle.load(fp)
         with open(f'model_ckpts/subj{all_subjs_F4[heldout_idx]}_un.pkl', 'wb') as fp:
            pickle.dump(best_checkpoint_data, fp)

# for i in range(len(subj_len)):
#     torch.set_default_dtype(torch.float64)
#     main(i,num_samples=1, gpus_per_trial=5)

# if train_losses:
#     np.save("train_losses",train_losses)
# if test_losses:
#     np.save("test_rmses",test_losses)
