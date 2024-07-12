
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import wandb
import torch

from sklearn.decomposition import PCA


def plot_pred_date(rpe,abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs,wandb_log=False):
    """
    abs_date : without the first seq_length
    """
    #fill train_predicted_rpe with nan at the end to have the same length as rpe
    train_predicted_rpe = np.concatenate((train_predicted_rpe,np.full(len(abs_date)-len(train_predicted_rpe),np.nan)))
                                
    test_predicted_rpe = np.concatenate((np.full(len(abs_date)-len(test_predicted_rpe),np.nan),test_predicted_rpe))


    df = pd.DataFrame()
    df['date'] = abs_date
    df['rpe'] = rpe
    df['train_predicted_rpe'] = train_predicted_rpe
    df['test_predicted_rpe'] = test_predicted_rpe
    df['train_diff'] = df['rpe'] - df['train_predicted_rpe']
    df['test_diff'] = df['rpe'] - df['test_predicted_rpe']

    #rolling mean
    df['train_diff'] = df['train_diff'].rolling(window=30,min_periods=seq_length//2).mean()
    df['test_diff'] = df['test_diff'].rolling(window=30,min_periods=seq_length//2).mean()

    # plot the difference between the true and predicted RPE
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x='date', y='train_diff', label='Train Difference',color='green')
    sns.lineplot(data=df, x='date', y='test_diff', label='Test Difference',color='red')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.legend()
    plt.title('Difference between True and Predicted RPE by month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lstm/diff_rpe_{}_{}.png'.format(seq_length,num_epochs))
    if wandb_log:
        wandb.log({"Difference between True and Predicted RPE by month": plt})
    plt.show()

    df['date'] = df['date'].dt.to_period('M')
    df['date'] = df['date'].dt.to_timestamp()
    df = df.groupby('date').mean().reset_index()


    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x='date', y='rpe', label='True RPE',color = 'blue')
    sns.lineplot(data=df, x='date', y='train_predicted_rpe', label='Train Predicted RPE',color='green')
    sns.lineplot(data=df, x='date', y='test_predicted_rpe', label='Test Predicted RPE',color='red')
    plt.xlabel('Date')
    plt.ylabel('RPE')
    plt.legend()
    plt.title('Predicted RPE vs True RPE by month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lstm/pred_rpe_{}_{}.png'.format(seq_length,num_epochs))
    if wandb_log:
        wandb.log({"Predicted RPE vs True RPE by month": plt})
    plt.show()

    



def plot_embedding(model, input_data,rpe,seq_length,num_epochs, labels=None, dim=2,name='PCA',wandb_log=False):
    # Step 1: Extract embedding data
    model.eval()

    with torch.no_grad():
        embedded_data = model.embedding(input_data).detach().cpu().numpy()
    
    # Step 2: Reduce dimensionality
    pca = PCA(n_components=dim)
    print(embedded_data.shape)
    reduced_data = pca.fit_transform(embedded_data)
    
    # Step 3: Create the plot
    fig = plt.figure(figsize=(10, 8))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=rpe)

    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=rpe)
    
    # Step 4: Add customization
    plt.title(f"Embedding Visualization {name}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
    if dim == 3:
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2f})")
    
    if rpe is not None:
        plt.colorbar(scatter, label="RPE")
    
    plt.tight_layout()
    plt.savefig('lstm/{}_embedding{}_{}.png'.format(name,seq_length,num_epochs))
    if wandb_log:
        wandb.log({f"Embedding Visualization {name}": plt})
    plt.show()


def plot_prediction(rpe,abs_date,train_predicted_rpe,test_predicted_rpe,seq_length,num_epochs,wandb_log=False):

    df = pd.DataFrame()
    df['date'] = abs_date
    df['rpe'] = rpe
    df['train_predicted_rpe'] = np.concatenate((train_predicted_rpe,np.full(len(abs_date)-len(train_predicted_rpe),np.nan)))
    df['test_predicted_rpe'] = np.concatenate((np.full(len(abs_date)-len(test_predicted_rpe),np.nan),test_predicted_rpe))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.lineplot(data=df, x='date', y='rpe', label='True RPE',color = 'blue',ax=ax)
    sns.lineplot(data=df, x='date', y='train_predicted_rpe', label='Train Predicted RPE',color='green',ax=ax)
    sns.lineplot(data=df, x='date', y='test_predicted_rpe', label='Test Predicted RPE',color='red',ax=ax)
    
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m\n%Y'))
    
    plt.xlabel('Date')
    plt.ylabel('RPE')
    plt.legend()
    plt.title('RPE Prediction')
    plt.savefig(f'lstm/rpe_prediction_v2_{seq_length}_seq_{num_epochs}_epochs.png')
    # plt.show()
    if wandb_log:
        wandb.log({"RPE Prediction": plt})
    plt.show()


def plot_pca(abs_date, hidden_data,training_len,test_len,seq_length,num_epochs,rpe, name, dim=2,wandb_log=False):


    hidden_data = hidden_data.permute(1,0,2).detach().cpu().numpy().reshape(training_len+test_len,-1)
    print(hidden_data.shape)
    
    abs_date_train = abs_date[:training_len]
    abs_date_test = abs_date[training_len:]

    hpca = PCA(n_components=4)
    hpca.fit(hidden_data)
    hvec_pca = hpca.transform(hidden_data)
    print(hpca.explained_variance_ratio_)

    train_df = pd.DataFrame()
    train_df['date'] = abs_date_train
    train_df['rpe'] = rpe[:training_len]
    train_df['pca0'] = hvec_pca[:training_len,0]
    train_df['pca1'] = hvec_pca[:training_len,1]

    test_df = pd.DataFrame()
    test_df['date'] = abs_date_test
    test_df['rpe'] = rpe[training_len:]
    test_df['pca0'] = hvec_pca[training_len:,0]
    test_df['pca1'] = hvec_pca[training_len:,1]


    #print pca based on abs
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.lineplot(data=train_df, x='date', y='pca0', label='PC1 Train',color='green',ax=ax)
    sns.lineplot(data=train_df, x='date', y='pca1', label='PC2 Train',color='blue',ax=ax)
    sns.lineplot(data=test_df, x='date', y='pca0', label='PC1 Test',color='red',ax=ax)
    sns.lineplot(data=test_df, x='date', y='pca1', label='PC2 Test',color='orange',ax=ax)
    ax.set_title(f'PCA of {name}')
    ax.legend()


    plt.savefig('lstm/pca_{}_{}_{}.png'.format(name,seq_length,num_epochs))
    if wandb_log:
        wandb.log({f"PCA {name} abs": plt})
    plt.show()

    
    #plot pca
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121)
    sc = ax.scatter(hvec_pca[:training_len,0],hvec_pca[:training_len,1],c=rpe[:training_len])
    ax.set_xlabel(f'PC1 {hpca.explained_variance_ratio_[0]:.2f}')
    ax.set_ylabel(f'PC2 {hpca.explained_variance_ratio_[1]:.2f}')
    plt.colorbar(sc,label='RPE')
    ax.set_title(f'PCA of {name} Train')
    ax = fig.add_subplot(122)
    sc= ax.scatter(hvec_pca[training_len:,0],hvec_pca[training_len:,1],c=rpe[training_len:])
    ax.set_xlabel(f'PC1 {hpca.explained_variance_ratio_[0]:.2f}')
    ax.set_ylabel(f'PC2 {hpca.explained_variance_ratio_[1]:.2f}')
    plt.colorbar(sc,label='RPE')
    ax.set_title(f'PCA of {name} Test')
    plt.savefig('lstm/pca_{}_{}_{}.png'.format(name,seq_length,num_epochs))
    
    if wandb_log:
        wandb.log({f"PCA {name}": plt})
    plt.show()
    