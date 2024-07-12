
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from ..script_hmm.hmm_script import predict_states
from ..utils.plot_utils import get_times_in_state

from tqdm import tqdm

from lstm_plot import *
from model import *


import locale
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

torch.manual_seed(0)
np.random.seed(0)
features = ['time_since_last_session', 'time_low', 'time_medium', 'time_high',
                                    'time_very_high', 'mean_power_low', 'mean_power_medium',
                                    'mean_power_high', 'mean_power_very_high']

## ----- INTERPRETABILITY -----
# - attention layer to see which time steps are important
# - hidden state visualization with PCA
# - hidden state visualization with t-SNE
# - hidden state visualization with UMAP
# - hidden state visualization with SHAP
# - hidden state visualization with LIME
# - Classique output / input or extra parameters correlation


#TODO : avec et sans normalization par PPR.
#TODO : avec et sans normalisation pat minmax   
#TODO : tester de garder les états du lstm entre les batchs. 
#TODO : garder les états h0, c0 par rapport à début de la séquence et sauvegarder les bons pour celle de sortie
#TODO : l'idéal c'est d'avoir des états partagé par time step et non par séquences.

def init_wandb():

    wandb.init(
        name="LSTM + resNet shared states + symb",
        # set the wandb project where this run will be logged
        project="LSTM",
        notes=
            # "on a un warmup de 100 epochs\n"
            "garder les états entre les batchs mais hidden unique par sequence\n"
            "on garde pas les états par time step\n"
            "ResNet + dropout"
            "watts reconstruct à partir de la ppr\n"
            "on modifie les états à + seq_length, pas de gradient pour h_vec\n"
            "lr scheduler cosine annealing warm restarts\n"
            "on apprend le dernier mlp sur le dernier h,c et le dernier input\n"
            # "pas de données de récupération\n"
            ,
        # track hyperparameters and run metadata
        config={
        "learning_rate_init": lr,
        "scheduler": lr_scheduler.__class__.__name__,
        "architecture": "LSTM",
        "dataset": "RPE",
        "return sequence": True,

        "seq_length": seq_length,
        "num_epochs": num_epochs,
        "hidden_size": hidden_size,
        "resnet_width": res_net_width,
        "embedding_dim": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
        "loss" : "MSELoss",
        "dropout": dropout,
        "regulizer": "L1Loss",
        "reg_hidden" : "MSELoss",
        "weight_loss": weight_loss
        }
    )

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y, index

def create_training_dataset(data_path,
                            norm_data_path,
                            meta_data_path,
                            model_path,
                            fields = ['stream_watts'],
                            label_states = ['low','medium','high','very high'],
                            palette = ['green','blue','orange','red'],
                            features = ['time_since_last_session', 'time_low', 'time_medium', 'time_high',
                                    'time_very_high', 'mean_power_low', 'mean_power_medium',
                                    'mean_power_high', 'mean_power_very_high'],
                            target = 'rpe'):

    norm_df = pd.read_pickle(data_path)
    norm_data_15600 = pd.read_pickle(norm_data_path)
    meta_data_15600 = pd.read_pickle(meta_data_path)
    with open(model_path,'rb') as f:
        model = pickle.load(f)

    norm_data_15600=norm_data_15600.merge(meta_data_15600[['id_session','poids']],how='inner',left_index=True,right_on='id_session')    
    predicted_15600 = predict_states(norm_df,model,fields,label_states,False)
    # print(norm_data_15600.head())

    predicted_15600 = predicted_15600.merge(norm_data_15600[['date','ppr','id_session']], how="inner", right_on='id_session', left_on="id_session")
    predicted_15600.loc[:,'stream_watts']= predicted_15600['stream_watts']*predicted_15600['ppr']

    # print(predicted_15600.head())
    #multiply by ppr


    times_in_state = get_times_in_state(predicted_15600,label_states,id_key='id_session')

    tps_per_session = predicted_15600.groupby('id_session').size()

    time_by_state = times_in_state.multiply(tps_per_session,axis=0)

    norm_data_15600.loc[:,'dt_date'] = pd.to_datetime(norm_data_15600['date'])
    norm_data_15600 = norm_data_15600.sort_values(by=['dt_date'])
    norm_data_15600['time_since_last_session'] = norm_data_15600['dt_date'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds() / 3600.0
    norm_data_15600['time_since_last_session'] = norm_data_15600['time_since_last_session'].fillna(0)

    #mean power by state and session
    mean_power = predicted_15600[['id_session','state_label','stream_watts']].groupby(['id_session','state_label']).mean().reset_index()
    mean_power = mean_power.pivot(index='id_session',columns='state_label',values='stream_watts')
    mean_power.fillna(0,inplace=True)

    mean_model = model.means_.reshape(-1)
    std_model = np.sqrt(model.covars_).reshape(-1)

    training_df = pd.DataFrame()
    training_df['id_session'] = norm_data_15600['id_session']
    training_df['date'] = pd.to_datetime(norm_data_15600['date'])
    training_df['poids'] = norm_data_15600['poids']
    training_df['time_since_last_session'] = norm_data_15600['time_since_last_session']
    training_df['rpe'] = norm_data_15600['rpe']
    training_df=training_df.merge(time_by_state,how='inner',left_on='id_session',right_index=True)
    training_df.rename(columns={'low':'time_low','medium':'time_medium','high':'time_high','very high':'time_very_high'},inplace=True)
    training_df=training_df.merge(mean_power[['low','medium','high','very high']],how='inner',left_on='id_session',right_index=True)


    training_df.rename(columns={'low':'mean_power_low','medium':'mean_power_medium','high':'mean_power_high','very high':'mean_power_very_high'},inplace=True)
    #div by poids
    training_df['mean_power_low'] = training_df['mean_power_low']/training_df['poids']
    training_df['mean_power_medium'] = training_df['mean_power_medium']/training_df['poids']
    training_df['mean_power_high'] = training_df['mean_power_high']/training_df['poids']
    training_df['mean_power_very_high'] = training_df['mean_power_very_high']/training_df['poids']

    training_df.drop(columns=['poids'],inplace=True)

    training_df.fillna(0,inplace=True)

    training_df=training_df.set_index('date')

    training_df[training_df['rpe']==0]=np.nan
    training_df.dropna(inplace=True)
    scaler = MinMaxScaler()
    training_df[features] = scaler.fit_transform(training_df[features])

    return training_df,mean_model,std_model





def create_sequences(df, features, target, seq_length,symb=False):
    sequences = []
    targets = []
    for i in range(len(df) - seq_length):
        seq = df[features].iloc[i:i+seq_length].values
        label = df[target].iloc[i+seq_length]
        sequences.append(seq)
        targets.append(label)
    if symb:
        return torch.from_numpy(np.array(sequences)).float(), torch.from_numpy(np.array(targets)).long()
    return torch.from_numpy(np.array(sequences)).float(), torch.from_numpy(np.array(targets)).float()

def prepare_loader(training_df,
                   ratio=0.8,
                   seq_length=100,
                   features = ['time_since_last_session', 'time_low', 'time_medium', 'time_high',
                                    'time_very_high', 'mean_power_low', 'mean_power_medium',
                                    'mean_power_high', 'mean_power_very_high'],
                     target = 'rpe',
                     symb=False):
    train_size =int(ratio * len(training_df)) 

    train_df = training_df.iloc[:train_size]
    # val_df = training_df.iloc[train_size-seq_length:train_size + val_size]
    test_df = training_df.iloc[train_size-seq_length:]



    X_train, y_train = create_sequences(train_df, features, target, seq_length,symb)
    # X_val, y_val = create_sequences(val_df, features, target, seq_length)
    X_test, y_test = create_sequences(test_df, features, target, seq_length,symb)
    train_dataset = CustomDataset(X_train, y_train)
    # print(train_dataset[0])
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    # val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_loader, test_loader,train_dataset,test_dataset



def train_warmpup(num_epochs,
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            regulizer,
            h_reg,
            c_reg,
            weight_loss,
            h_vec,
            c_vec,
            lr_scheduler):
    epo = tqdm(range(num_epochs))
    h_bloss = 0
    c_bloss = 0
    mse_bloss = 0
    reg_bloss = 0
    for epoch in epo:
        model.train()
        for X_batch, y_batch,indices in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            h = h_vec[:, indices, :]
            c = c_vec[:, indices, :]
            
            recup_batch = X_batch[:,-1,0]
            X_batch = X_batch[:,:,1:]
            outputs,h_next,c_next = model(X_batch,h,c,recup_batch)

            mse_loss = criterion(outputs, y_batch)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += regulizer(param, torch.zeros_like(param))
            # h_loss = h_reg(h_next, h_vec[:, indices + seq_length, :])
            # c_loss = c_reg(c_next, c_vec[:, indices + seq_length, :])

            loss = (mse_loss*weight_loss["loss"]
                    +  weight_loss["regulizer"] * reg_loss
                    # + weight_loss["reg_h"] * h_loss
                    # + weight_loss["reg_c"] * c_loss)
                    )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     h_vec[:, indices + seq_length, :] = h_next.detach().clone().requires_grad_(False)
            #     c_vec[:, indices + seq_length, :] = c_next.detach().clone().requires_grad_(False)
            # h_bloss += h_loss.item()
            # c_bloss += c_loss.item()
            mse_bloss += mse_loss.item()
            reg_bloss += reg_loss.item()
        # h_bloss /= len(train_loader)
        # c_bloss /= len(train_loader)
        mse_bloss /= len(train_loader)
        reg_bloss /= len(train_loader)
        lr_scheduler.step()






def train(num_epochs, 
          model, 
          train_loader, 
          test_loader, 
          criterion, 
          optimizer, 
          regulizer, 
          h_reg, 
          c_reg, 
          weight_loss, 
          h_vec, 
          c_vec, 
        #   all_h_res,
        #   all_c_res,
          lr_scheduler):
    epo = tqdm(range(num_epochs))
 
    for epoch in epo:
        model.train()
        h_bloss = 0
        c_bloss = 0
        mse_bloss = 0
        reg_bloss = 0
        for X_batch, y_batch,indices in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            h = h_vec[:, indices, :]
            c = c_vec[:, indices, :]
            
            outputs,h_next,c_next = model(X_batch,h,c)#,recup_batch)#h_pred,c_pred

            mse_loss = criterion(outputs.squeeze(), y_batch)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += regulizer(param, torch.zeros_like(param))
            h_loss = h_reg(h_next, h_vec[:, indices + seq_length-1, :]) 
            c_loss = c_reg(c_next, c_vec[:, indices + seq_length-1, :])
            
            loss = (mse_loss*weight_loss["loss"]
                    +  weight_loss["regulizer"] * reg_loss)
                    #  + weight_loss["reg_h"] * h_loss
                    #  + weight_loss["reg_c"] * c_loss)
                    # + weight_loss["dist_loss"] * dist_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                h_vec[:, indices + seq_length-1, :] = h_next
                c_vec[:, indices + seq_length-1, :] = c_next
            h_bloss += h_loss.item()
            c_bloss += c_loss.item()
            mse_bloss += mse_loss.item()
            reg_bloss += reg_loss.item()
        h_bloss /= len(train_loader)
        c_bloss /= len(train_loader)
        mse_bloss /= len(train_loader)
        reg_bloss /= len(train_loader)
        lr_scheduler.step()
        
        model.eval()
        val_loss = 0
        h_vec_tmp = h_vec.clone()
        c_vec_tmp = c_vec.clone()
        # print(h_vec_tmp.shape)

        with torch.no_grad():
            for X_batch, y_batch,indices in test_loader:
                indices = indices + len(train_dataset)
                # print(indices)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                h= h_vec_tmp[:,indices,:]
                c= c_vec_tmp[:,indices,:]
                outputs,h_next,c_next = model(X_batch,h,c)#,recup_batch)#h_pred,c_pred)
                
                loss = criterion(outputs.squeeze(), y_batch)
                # remove the indices that are > h_vec.shape[1]
                indices = indices[indices < h_vec.shape[1] - seq_length + 1]
                h_vec_tmp[:,indices+seq_length-1,:] = h_next[:,:indices.shape[0],:]#(h_res + all_h_res[:,indices+seq_length,:])/2
                h_vec_tmp[:,indices+seq_length-1,:] = c_next[:,:indices.shape[0],:]#(c_res + all_c_res[:,indices+seq_length,:])/2

                val_loss += loss.item()
        val_loss /= len(test_loader)

        if wandb_log:
            wandb.log({"Training Loss": mse_bloss,
                    "Regulizer Loss": reg_bloss,
                    "Validation Loss": val_loss,
                    "Regulizer h": h_bloss,
                    "Regulizer c": c_bloss,
                    'lr' : optimizer.param_groups[0]['lr']
                        })

def test_model(model, train_dataset,test_dataset,criterion,h_vec,c_vec):#,all_h_res, all_c_res):
    train_predicted_rpe = []
    val_predicted_rpe = []
    test_predicted_rpe = []
    model.eval()
    test_loss = 0
    train_loss = 0
    all_y_train = []
    all_y_test = []
    with torch.no_grad():
        for X_train,y_train,indices in train_dataset:
            X_train = X_train.to(device).unsqueeze(0)
            y_train = y_train.to(device).unsqueeze(0)

            h= h_vec[:,indices,:].reshape(num_layers, 1, hidden_size)
            c= c_vec[:,indices,:].reshape(num_layers, 1, hidden_size)

            outputs,h_next,c_next = model(X_train,h,c)#,recup_batch)#h_pred,c_pred)
            loss = criterion(outputs.squeeze(0), y_train)
            h_vec[:,indices+seq_length-1,:] = h_next.squeeze()#(h_res + all_h_res[:,indices+seq_length,:])/2
            c_vec[:,indices+seq_length-1,:] = c_next.squeeze()#(c_res + all_c_res[:,indices+seq_length,:])/2

            prediction = torch.argmax(outputs.squeeze(0))
            train_predicted_rpe.append(prediction.to('cpu').numpy())
            all_y_train.append(y_train.to('cpu').numpy())
            
            train_loss += loss.item()
        train_loss /= len(train_dataset)
            
        for X_test,y_test,indices in test_dataset:

            X_test = X_test.to(device).unsqueeze(0)
            y_test = y_test.to(device).unsqueeze(0)
            indices = indices + len(train_dataset)
            h= h_vec[:,indices,:].reshape(num_layers, 1, hidden_size).contiguous()
            c= c_vec[:,indices,:].reshape(num_layers, 1, hidden_size).contiguous()

            outputs,h_next,c_next = model(X_test,h,c)#,recup_batch)#h_pred,c_pred)
            loss = criterion(outputs.squeeze(0), y_test)
            if indices< h_vec.shape[1]-seq_length+1:
                h_vec[:,indices+seq_length-1,:] = h_next.squeeze() #(h_res.reshape(num_layers,hidden_size) + all_h_res[:,indices+seq_length,:])/2
                c_vec[:,indices+seq_length-1,:] = c_next.squeeze() #(c_res.reshape(num_layers,hidden_size) + all_c_res[:,indices+seq_length,:])/2
            
            prediction = torch.argmax(outputs.squeeze(0))
            test_predicted_rpe.append(prediction.to('cpu').numpy())
            test_loss += loss.item()
            all_y_test.append(y_test.to('cpu').numpy())
        test_loss /= len(test_dataset)

    train_predicted_rpe = np.array(train_predicted_rpe).reshape(-1)
    test_predicted_rpe = np.array(test_predicted_rpe).reshape(-1)

    y_train = np.array(all_y_train).reshape(-1)
    y_test = np.array(all_y_test).reshape(-1)

    #compute MAPE
    mape_train = np.mean(np.abs(y_train - train_predicted_rpe) / y_train)
    mape_test = np.mean(np.abs(y_test - test_predicted_rpe) / y_test)
    #Compute WAPE
    wape_train = np.sum(np.abs(y_train - train_predicted_rpe)) / np.sum(y_train)
    wape_test = np.sum(np.abs(y_test - test_predicted_rpe)) / np.sum(y_test)
    accuracy_train = np.sum(y_train == train_predicted_rpe) / len(y_train)
    accuracy_test = np.sum(y_test == test_predicted_rpe) / len(y_test)
    return train_predicted_rpe,val_predicted_rpe,test_predicted_rpe,mape_train,mape_test,train_loss,test_loss,wape_train,wape_test,accuracy_train,accuracy_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a LSTM model to predict RPE')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for the LSTM')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('-n', '--num_epochs', type=int, default=10000, help='Number of epochs')
    args = parser.parse_args()
    # Load the data
    #Test with normalized power.

    data_path = 'data/normalized_rolling_15600.pkl'
    norm_data_path = 'data/norm_data_15600.pkl'
    meta_data_path = 'data/meta_15600.pkl'
    model_path = 'model_final_15600/power/hmm_4comp_1mix_500_diag_norm_roll_prior_mean.pkl'

    seq_length = args.seq_length
    wandb_log = args.wandb
    num_epochs = args.num_epochs
    num_warmup = num_epochs//10
    # Calculate indices for splitting
    input_size = 9 #train_loader.dataset.shape()[2]
    hidden_size = 256
    res_net_width = 32 
    # embedding_dim = 4
    num_layers = 1
    is_bidirectional = False
    num_layers_resnet = 10
    dropout = 0.5
    output_size = 10
    lr = 0.0007
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_loss = {
        "reg_h": 0,
        "reg_c": 0,
        "regulizer": 0.0001,
        "loss": 1,
        "dist_loss": 0
    }


    training_df ,mean_model,std_model= create_training_dataset(data_path,norm_data_path,meta_data_path,model_path)

    train_loader, test_loader,train_dataset,test_dataset = prepare_loader(training_df, seq_length=seq_length,symb=True)

    # h_res = torch.zeros(num_layers, len(train_dataset)+len(test_dataset), hidden_size).to(device).requires_grad_(False)
    # c_res = torch.zeros(num_layers,len(train_dataset)+len(test_dataset), hidden_size).to(device).requires_grad_(False)

    model = RPE_LSTM_h_symb(input_size,
                     hidden_size, 
                        res_net_width,
                     num_layers_resnet,
                     output_size,
                    dropout=dropout).to(device)
    # criterion = nn.MSELoss()

    #load the model
    encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, 
                       num_layers=1, batch_first=True,dropout=dropout,
                       bidirectional=is_bidirectional).to(device)

    decoder = ResNet_decoder(hidden_size, output_size,dropout=dropout,num_layers=num_layers_resnet,res_net_width=res_net_width).to(device)
    
    model = Encoder_Decoder(encoder,decoder,input_size,hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    regulizer = nn.L1Loss()
    h_reg = nn.MSELoss()
    c_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=lr, step_size_up=100, gamma=0.8)
    #scheduler cosinus with exp decay
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998)
    
    h_vec = torch.zeros(num_layers, len(train_dataset)+len(test_dataset), hidden_size).to(device).requires_grad_(False)
    c_vec = torch.zeros(num_layers,len(train_dataset)+len(test_dataset), hidden_size).to(device).requires_grad_(False)

    if wandb_log:
        init_wandb()

    # train_warmpup(num_warmup,
    #         model, 
    #         train_loader, 
    #         test_loader, 
    #         criterion, 
    #         optimizer, 
    #         regulizer, 
    #         h_reg, 
    #         c_reg, 
    #         weight_loss, 
    #         h_vec, 
    #         c_vec, 
    #         lr_scheduler)
    try:
        train(num_epochs,
            model, 
            train_loader, 
            test_loader, 
            criterion, 
            optimizer, 
            regulizer, 
            h_reg, 
            c_reg, 
            weight_loss, 
            h_vec, 
            c_vec, 
            # h_res,
            # c_res,
            lr_scheduler)
                # Save the model
        os.makedirs('lstm', exist_ok=True)
        torch.save(model.state_dict(), 'lstm/rpe_lstm.pth')
        rpe = training_df['rpe'].values
        abs_date = training_df.index[seq_length:]

        plot_embedding(model, torch.tensor(training_df[features].values[:int(len(training_df)*0.8),:],dtype=torch.float).to(device),
                       rpe[:int(len(training_df)*0.8)],seq_length,num_epochs, dim=2,name="train",wandb_log=wandb_log)

        plot_embedding(model, torch.tensor(training_df[features].values[int(len(training_df)*0.8)-seq_length:,:],dtype=torch.float).to(device), 
                    rpe[int(len(training_df)*0.8)-seq_length:],
                    seq_length,num_epochs,dim=2,name="test",wandb_log=wandb_log)

        (train_predicted_rpe,
        val_predicted_rpe,
        test_predicted_rpe,
        mape_train,
        mape_test, 
        train_loss,
        test_loss,
        wape_train,
        wape_test,
        accuracy_train,
        accuracy_test
        ) = test_model(model, train_dataset,test_dataset,criterion,h_vec,c_vec)#,h_res,c_res)
        # abs_train = np.arange(len(train_predicted_rpe))+seq_length
        # abs_test = np.arange(len(test_predicted_rpe))+seq_length+len(train_predicted_rpe)

        # print(abs_train.shape,train_predicted_rpe.shape)
        # print(abs_test.shape,test_predicted_rpe.shape)

        plot_prediction(rpe[seq_length:],abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs, wandb_log)
        plot_pred_date(rpe[seq_length:],abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs,wandb_log=wandb_log)

        plot_pca(abs_date, h_vec,len(train_dataset),len(test_dataset),seq_length,num_epochs,rpe[seq_length:],'h_vec', dim=2,wandb_log=wandb_log)

        plot_pca(abs_date, c_vec,len(train_dataset),len(test_dataset),seq_length,num_epochs,rpe[seq_length:],'c_vec', dim=2,wandb_log=wandb_log)
        

        if wandb_log:  
            wandb.log({
                "Test Loss final": test_loss,
                "Train Loss final": train_loss,
                "MAPE train": mape_train,
                # "WMAPE val": wmape_val,
                "MAPE test": mape_test,
                "WMAPE train": wape_train,
                "WMAPE test": wape_test,
                "Accuracy train": accuracy_train,
                "Accuracy test": accuracy_test,
                # "RPE Prediction": plt
                })
            wandb.finish()
    except KeyboardInterrupt:  
        # Save the model
        os.makedirs('lstm', exist_ok=True)
        torch.save(model.state_dict(), 'lstm/rpe_lstm.pth')
        rpe = training_df['rpe'].values

        plot_embedding(model, torch.tensor(training_df[features].values[:int(len(training_df)*0.8),:],dtype=torch.float).to(device),
                       rpe[:int(len(training_df)*0.8)],seq_length,num_epochs, dim=2,name="train",wandb_log=wandb_log)

        plot_embedding(model, torch.tensor(training_df[features].values[int(len(training_df)*0.8)-seq_length:,:],dtype=torch.float).to(device), 
                    rpe[int(len(training_df)*0.8)-seq_length:],
                    seq_length,num_epochs,dim=2,name="test",wandb_log=wandb_log)

        (train_predicted_rpe,
        val_predicted_rpe,
        test_predicted_rpe,
        mape_train,
        mape_test, 
        train_loss,
        test_loss,
        wape_train,
        wape_test,
        accuracy_train,
        accuracy_test
        ) = test_model(model, train_dataset,test_dataset,criterion,h_vec,c_vec)#,h_res,c_res)
        abs_train = np.arange(len(train_predicted_rpe))+seq_length
        abs_test = np.arange(len(test_predicted_rpe))+seq_length+len(train_predicted_rpe)


        # rpe = training_df['rpe'].values
        abs_date = training_df.index[seq_length:]
        plot_prediction(rpe[seq_length:],abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs, wandb_log)
        plot_pred_date(rpe[seq_length:],abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs,wandb_log=wandb_log)

        
        plot_pca(abs_date, h_vec,len(train_dataset),len(test_dataset),seq_length,num_epochs,rpe[seq_length:],'h_vec', dim=2,wandb_log=wandb_log)

        plot_pca(abs_date, c_vec,len(train_dataset),len(test_dataset),seq_length,num_epochs,rpe[seq_length:],'c_vec', dim=2,wandb_log=wandb_log)
        

        if wandb_log:  
            wandb.log({
                "Test Loss final": test_loss,
                "Train Loss final": train_loss,
                "MAPE train": mape_train,
                # "WMAPE val": wmape_val,
                "MAPE test": mape_test,
                "WMAPE train": wape_train,
                "WMAPE test": wape_test,
                "Accuracy train": accuracy_train,
                "Accuracy test": accuracy_test,
                # "RPE Prediction": plt
                })
            wandb.finish()
        

#TODO la loss hidden fonctionne pas les états cachés convergent vers 0...