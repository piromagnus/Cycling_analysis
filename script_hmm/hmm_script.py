
import pickle
from cv2 import norm
from matplotlib import figure
from mim import train
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import plot
from tqdm import tqdm
import numpy as np
from hmmlearn import hmm
import os
import argparse

from ..utils.transformation import cleaning,normalize_data,normalize_all_ath,session_id_to_ath_id
from ..utils.plot_utils import *
from ..utils.io import load_data,write_model

sns.set_theme()
np.random.seed(1)

def trainHmm(df:pd.DataFrame,
             data_fields=["stream_watts"],
             nb_components=3,
             nb_mix=1,
             covariance_type="diag",
             n_iter=300,
             verbose=True,
             variation=False,
             ma_window=120,
             id_key="id"):
    

    #plot histogram of the normalized data to check if it is normal only for stream_watts
    # sns.histplot(data=normalized_df,x="stream_watts",kde=True)
    # plt.title("Histogram of the normalized data")
    # plt.show()
    keys= [id_key,"tps"]+data_fields
    # print(df[keys].head())
    # print(normalized_df[keys])
    # moving_average = df[keys].groupby(id_key).rolling(ma_window,center=True,min_periods=1).mean().reset_index()
    # print([group[data_fields].values.shape for session_id, group in moving_average.groupby('id')])
    # deriv_sessions = [np.gradient(group[data_fields].values.reshape(-1)) for session_id, group in moving_average.groupby(id_key)]

    # get the sessions but keep the order
    sessions = [group[data_fields].values for session_id, group in df.groupby(id_key)]
    id_order = df[id_key].unique()
    id_sessions = [id_session for id_session, group in df.groupby(id_key)]
    
    #reorder
    ids,sessions = reorder_sessions(sessions, id_sessions, id_order)


    #print the shape of the sessions
    # small_group = df[id_key].map(df.groupby(id_key).size()<100)
    # print(df[small_group].groupby(id_key).size())
    print("sessions created")

    # Create the HMM model with Gaussian emissions and 4 states

    # use a prior on the transition matrix
    prior = 4*np.eye(nb_components) + 1
        
    if nb_mix == 1:
        model_full = hmm.GaussianHMM(n_components=nb_components, covariance_type=covariance_type, n_iter=n_iter,verbose=verbose,transmat_prior=prior,init_params='')
    else:
        model_full = hmm.GMMHMM(n_components=nb_components,n_mix=nb_mix, covariance_type=covariance_type, n_iter=n_iter,verbose=verbose,transmat_prior=prior,init_params='')
   
    # Init with diagonal identity transition matrix
    startprob, transmat, means, covars = init_hmm(df[data_fields],nb_components,covariance_type)
    model_full.startprob_ = startprob
    model_full.transmat_ = transmat
    model_full.means_ = means
    model_full.covars_ = covars
   
    # Concatenate all sessions for training
    # if variation:
    #     X = np.concatenate(deriv_sessions).reshape(-1,len(data_fields))
    #     lengths = [len(session) for session in deriv_sessions]
    # else:
    X = np.concatenate(sessions).reshape(-1,len(data_fields))
    lengths = [len(session) for session in sessions] 


    # Train the HMM
    model_full.fit(X,lengths=lengths)
    print("model trained")
    return model_full

def reorder_sessions(sessions, id_sessions, id_order):
    # Create a mapping from id_order to its index position
    id_order_map = {id: i for i, id in enumerate(id_order)}
    
    # Create a list of tuples containing id_session and corresponding session
    combined = list(zip(id_sessions, sessions))
    
    # Sort the combined list based on the order specified in id_order_map
    combined_sorted = sorted(combined, key=lambda x: id_order_map[x[0]])
    
    # Unzip the sorted list back into id_sessions and sessions
    id_sessions_sorted, sessions_sorted = zip(*combined_sorted)
    
    return list(id_sessions_sorted), list(sessions_sorted)


def predict_states(cleaned:pd.DataFrame,model:hmm.BaseHMM,data_field=["stream_watts"],label_states=["low","medium","high"],variation=False,id_key="id_session"):

    print(cleaned.head())
    id_order = cleaned[id_key].unique()
    id_sessions = [id_session for id_session, group in cleaned.groupby(id_key)]

    if variation:
        keys= data_field+[id_key]+["tps"]
        moving_average = cleaned[keys].groupby(id_key).rolling(120,center=True,min_periods=1).mean().reset_index()
        deriv_sessions = [np.gradient(group[data_field].values.reshape(-1)) for session_id, group in moving_average.groupby(id_key)]
        sessions = [group[data_field].values for session_id, group in cleaned.groupby(id_key)]
        segmented_sessions = [model.predict(session.reshape(-1, len(data_field))) for session in deriv_sessions]
    else:
        sessions = [group[data_field].values for session_id, group in cleaned.groupby(id_key)]
        ids,sessions = reorder_sessions(sessions, id_sessions, id_order)
        segmented_sessions = [model.predict(session.reshape(-1, len(data_field))) for session in sessions]

    

    # Add predicted states to the dataframe
    predicted_states = np.concatenate(segmented_sessions)
    new_cleaned = cleaned.copy()
    new_cleaned['state'] = predicted_states
    # state_68 = new_cleaned[["stream_watts","state"]].groupby('state').mean()+new_cleaned[["stream_watts","state"]].groupby('state').std()
    # print(state_68)
    # state_68 = state_68.sort_values(by="stream_watts")
    means= model.means_
    order = np.argsort(means[:,0])
    map_states_order = {order[i]:i for i in range(len(order))}

    map_states_label = {order[i]:label_states[i] for i in range(len(order))}
    #create a mapping to permute the states id

    
    # print(map_states_label)
    # print(order)

    new_cleaned["state_label"] = new_cleaned["state"].map(map_states_label)
    new_cleaned["state"] = new_cleaned["state"].map(map_states_order)
    # print(new_cleaned.head())
    return new_cleaned

def init_hmm(data:pd.DataFrame,n_components:int,covariance_type:str):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, n_init=3, init_params='kmeans',verbose=1)
    gmm.fit(data)

    # Use GMM parameters to initialize the HMM
    sticky_factor = 0.8
    startprob = np.full(n_components, 1 / n_components)
    transmat = np.full((n_components, n_components), (1 - sticky_factor) / (n_components - 1)) 
    np.fill_diagonal(transmat, sticky_factor)
    means = gmm.means_
    covars = gmm.covariances_
    return startprob, transmat, means, covars

if __name__ == "__main__":

    # Put the full name with -- and the minimal name (1 letter) with -

    parser = argparse.ArgumentParser(description='Train a HMM model on the data')
    parser.add_argument('--nComp', type=int, default=3, help='Number of states in the HMM')
    parser.add_argument('--nMix', type=int, default=1, help='Number of mixtures in the GMM')
    parser.add_argument('--covType', type=str, default='diag', help='Type of covariance matrix')
    parser.add_argument('--nIter', type=int, default=300, help='Number of iterations for the training')
    # store true or false
    parser.add_argument('-v,--variation', dest='variation', action='store_true', help='Use the variation of the data')
    parser.add_argument('-n,--normalized', dest='normalized', action='store_true', help='Normalize the data')
    # multiple values
    parser.add_argument('--dataField', type=str, nargs='+', default=["stream_watts"], help='Fields to use for the training')
    parser.add_argument('-s,--save', dest='save', action='store_true', help='Save the model')
    parser.add_argument('-r,--rolling', dest='rolling', action='store_true', help='Use the rolling mean')
    parser.add_argument('-p,--prior', dest='prior', action='store_true', help='Use the initial data')
    parser.add_argument('--method_power', type=str, default='ppr30', help='Method to use for the power normalization')
    parser.add_argument('--method_hr', type=str, default='mean', help='Method to use for the heart rate normalization')
    # Add flag that can be concatenated for the plots
    # e for emission matrix, t for time spent in each state, r for random sessions, m for mean power for each state

    
    parser.add_argument('--plot', type=str, default='retm', help='Plot the data')
    args = parser.parse_args()

    folder_fields = "_".join(["hr" if "heartrate" in i else "power" for i in args.dataField])
    template = "{}comp_{}mix_{}_{}{}{}{}{}{}".format(args.nComp,args.nMix,args.nIter,args.covType,
                                                    "_var" if args.variation else "",
                                                    "_norm" if args.normalized else "",
                                                    "_roll" if args.rolling else "",
                                                    "_prior" if args.prior else "",
                                                    "_mean" if args.method_hr == "mean" else "_max")
    figure_path = "hmm_final/"+folder_fields+"/"
    os.makedirs(figure_path,exist_ok=True)
    model_dir = "model_final/"+folder_fields+"/"
    model_path = model_dir+"hmm_"+template+".pkl"
    normalize_data_path = "data/all_normalized{}.pkl".format("_rolling" if args.rolling else "")
    meta_path = "data/meta_data.pkl"

    emission_path = figure_path+"emission_matrix_"+template+".png"
    emission_box_path = figure_path+"emission_boxplot_"+template+".png"
    random_path = figure_path+"random_sessions_"+template
    time_path = figure_path+"time_spent_in_each_state_"+template+".png"
    mean_path = figure_path+"mean_power_"+template+".png"
    matrix_path = figure_path+"matrix_transition_"+template
    score_path = figure_path+"scores_"+template+".txt"

    print("Running the script with the following parameters : ")
    print("Number of components : ",args.nComp)
    print("Number of mixtures : ",args.nMix)
    print("Covariance type : ",args.covType)
    print("Number of iterations : ",args.nIter)
    print("Variation : ",args.variation)
    print("Normalized : ",args.normalized)
    print("Data fields : ",args.dataField)
    print("Rolling : ",args.rolling)
    print("Prior : ",args.prior)
    print("Method power : ",args.method_power)
    print("Method heart rate : ",args.method_hr)
    print("Plot : ",args.plot)


    if args.nComp == 3:
        label_states = ["low","medium","high"]
        palette = ['green','blue','red']
    elif args.nComp == 4:
        label_states = ["low","medium","high","very high"]
        palette = ['green','blue','orange','red']
    elif args.nComp == 5:
        label_states = ["rest","low","medium","high","extreme"]
        palette = ['green','blue','purple','orange','red']
    else:
        label_states = ["State_{}".format(i) for i in range(args.nComp)]
        palette = ['green','blue','purple','yellow','orange','pink','red','brown','grey','black']


    # cleaned_data_path = "data/cleaned_sorted.pkl"

    # full,rpe = load_data()
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as handle:
            meta_data = pickle.load(handle)

    # if os.path.exists(cleaned_data_path):
    #     print("loading cleaned data")
    #     cleaned = pd.read_pickle(cleaned_data_path)

    # else:
    #     print("cleaning data")
    #     cleaned = cleaning(full,rpe)
    #     os.makedirs("data",exist_ok=True)
    #     cleaned.to_pickle(cleaned_data_path)
    #     print("data cleaned")
    
    # if args.normalized:
    #     if os.path.exists(normalize_data_path):
    #         print("loading normalized data")
    #         training_df = pd.read_pickle(normalize_data_path)
    #     else:
    #         print("before normalization : ",cleaned.shape)
    #         if "ath_id" in cleaned.columns:
    #             training_df,norm_data= normalize_all_ath(cleaned,
    #                                            meta_data,
    #                                            fields=args.dataField,
    #                                            period=60,
    #                                            rolling=args.rolling,
    #                                            method_hr = args.method_hr,
    #                                            id_key="session_id")
    #             print("after normalization : ",training_df.shape)
    #             norm_data.to_pickle("data/norm_data.pkl")
    #             training_df = training_df.dropna(subset=["stream_watts","stream_heartrate"])
    #         else:
    #             training_df = normalize_data(cleaned,rpe,fields=["stream_heartrate","stream_watts"],period=60,rolling=args.rolling,method_hr = args.method_hr)
    #             print("after normalization : ",training_df.shape)

    #             training_df = pd.merge(training_df,rpe[["id_session","dt_session","rpe"]],
    #                                 left_on="id",right_on="id_session",how="left")
    #             print("after merge : ",training_df.shape)

    #             training_df = training_df.dropna(subset=["stream_watts","stream_heartrate"])
            
    #         # print(training_df)
            
    #         training_df.to_pickle(normalize_data_path)
        # print(training_df)

    training_df = pd.read_pickle(normalize_data_path)
    # print(training_df.head())
    training_df = training_df.dropna(subset=args.dataField)
    meta_data = pd.read_pickle(meta_path)


    if 'ath_id' in training_df.columns:
        id_key = "id_session"
    else:
        id_key = "id"
    # model = trainHmm(training_df,
    #                  data_fields=args.dataField,
    #                  nb_components=args.nComp,
    #                  nb_mix=args.nMix,
    #                  covariance_type=args.covType,
    #                  n_iter=args.nIter,
    #                  verbose=True,
    #                  variation=args.variation,
    #                  id_key=id_key)

    
    model = pd.read_pickle(model_path)
    
    if args.save:
        os.makedirs(model_dir,exist_ok=True)
        write_model(model,model_path)
        
    scores_model(model,training_df[args.dataField].values,training_df.groupby(id_key).size().values,score_path)


    predicted = predict_states(training_df,model,data_field=args.dataField,label_states=label_states,variation=args.variation,id_key=id_key)
    
    generate_tikz_hmm(model, 
                      matrix_path,
                      label_states=label_states,
                      colors=palette)
    
    if 't' in args.plot:
        plot_times_in_state(predicted,meta_data,
                            time_path,
                            label_states=label_states,
                            palette=palette,
                            id_key=id_key)
    if 'r' in args.plot:
        plot_random_sessions(predicted,
                            random_path,
                            label_states=label_states,
                            data_fields=args.dataField,
                            palette=palette,
                            id_key=id_key)
    if 'e' in args.plot:
        if args.nMix == 1:
            if args.covType =='diag':
                plot_emission_matrix(model,
                                    emission_path,
                                    label_states=label_states,
                                    data_fields=args.dataField,
                                    palette=palette)
                plot_gaussian_emission_boxplot(model,emission_box_path,
                                            label_states=label_states,
                                                data_fields=args.dataField,
                                                palette=palette)
            else:
                print("plotting emission matrix for full cov")
                plot_gaussian_full_emission_boxplot(model,
                                    emission_path,
                                    label_states=label_states,
                                    data_fields=args.dataField,
                                    palette=palette)
        else:

            plot_gmm_emission_matrix(model,
                                    emission_path,
                                    label_states=label_states,
                                    data_fields=args.dataField,
                                    palette=palette)
   
    if 'm' in args.plot:
        print("bug here, i don't know why so i wont appear")
        # plot_means_states(predicted,mean_template.format(args.nComp,args.nMix,args.nIter,args.covType,"var" if args.variation else "no_var","norm" if args.normalized else "no_norm"),label_states=label_states)
    

    #TODO plot the power for each state of the HR.
    print("done")
    

