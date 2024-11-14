from training_helper import prep_data, circular_hist
import time
import numpy as np
import torch
import torch.nn as nn
import gc
import torch.optim as optim
import torch.utils.data as datas
import matplotlib.pyplot as plt
from pathlib import Path
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
old_subjs_F4 = ['175','158','167','165','122','107','157','150','146','104','135'] #Done
young_subjs_F4 = ['154', '156', '177', '132', '131', '171', '174']
all_subjs_F4 = old_subjs_F4 + young_subjs_F4
#tests_F4_old = ['143','152',]
#tests_F4_young = ['159', '114']

X_data, Y_data, subj_len = get_data(all_subjs_F4)

torch.set_default_dtype(torch.float64)

class GRU(nn.Module):
    '''
    layers = [
    gruLayer(16,"Name","gru_1")
    gruLayer(4,"Name","gru_2")
    fullyConnectedLayer(2,"Name","fc")
    '''
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
        device = x.device
        h1 = self.h1.to(device) # make sure ur training using a gpu or else there will be an error
        h2 = self.h2.to(device)
        # h1 = self.h1
        # h2 = self.h2
        x,h1 = self.gru1(x,h1)
        x,h2 = self.gru2(x,h2)
        x = self.fc(x)
        self.h1 = h1.detach()
        self.h2 = h2.detach()
        del h1
        del h2
        return x
 
def gen_nfold_subjs(val_subj_idx,subj_len,X_train,Y_train):
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


def train_model(subj_idx,epoch,batch_size,lr = 0.001,save_ckpt = True):
    # initialize model param
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GRU()
    net.init_weights()
    net.to(device)
    criterion = nn.MSELoss(reduction = "mean")
    optimizer = optim.Adam(net.parameters(),lr=lr)

    # get data
    train_x,train_y,test_x,test_y = gen_nfold_subjs(subj_idx,subj_len,X_data, Y_data)

    trainset = datas.TensorDataset(train_x,train_y)
    trainloader = datas.DataLoader(
        trainset, batch_size=batch_size, shuffle=False)
    
    testset = datas.TensorDataset(test_x,test_y)
    testloader = datas.DataLoader(
        testset, batch_size=batch_size, shuffle=True)
    
    # train model
    print(f"Subject {all_subjs_F4[subj_idx]} training started.")
    start_time = time.time()
    test_loss = []
    for i in range(epoch):
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
            del inputs
            del targets                
            gc.collect()
            torch.cuda.empty_cache()
        # train loss
        net.eval()
        with torch.no_grad():
            test_rmse = 0
            for inputs, targets in testloader:
                # test output
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                test_rmse += np.sqrt(criterion(outputs, targets).item())
                del inputs
                del targets
                gc.collect()
                torch.cuda.empty_cache()
            test_loss.append(test_rmse**2)
            print(f"Epoch {i} Test Loss: {test_rmse}")
    
    end_time = time.time()
    train_time = (end_time - start_time)/60 #mins
    print(f"Subject {all_subjs_F4[subj_idx]} training finished in {train_time} mins.")

    # save checkpoint
    if save_ckpt:
        data_path = f"model_ckpts/subj{all_subjs_F4[subj_idx]}.pkl"
        checkpoint_data = {
            "epoch": epoch,
            "batch_size": batch_size,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": test_rmse**2
        }
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)
    return test_loss

####uncoment to train

# for i in range(1):
#     test_loss = train_model(i, epoch = 10,batch_size = 64)
#     test_rmses.append(test_loss)

# test_rmses = []
# if test_rmses:
#     np.save("test_rmses",test_rmses )

