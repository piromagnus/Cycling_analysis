import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from hmmlearn import hmm
import os
from ..utils.normalizer import AthleteTransformer
import argparse
from ..utils.plot_utils import *
from ..utils.transformation import cleaning
from hmm_script import predict_states


def load_data():
    with open('df_clust_all.pkl', 'rb') as handle:
        full = pickle.load(handle)
        
    with open('df_rpe.pkl', 'rb') as handle:
        rpe = pickle.load(handle)
    return full,rpe


def plot_all_figures(n_comp,n_mix,n_iter,mat,var,norm,rolling,prior,fields):
    
    
    folder_fields = "_".join(["hr" if "heartrate" in i else "power" for i in fields])
    label_states = ["low","medium","high"] if n_comp == 3 else ( ["low","medium","high","very high"] if n_comp == 4 else ["low","medium","high","very high","extreme"])
    palette = ['green','blue','red'] if n_comp == 3 else (['green','blue','orange','red'] if n_comp == 4 else ['green','blue','purple','orange','red'])

    figure_savedir = "hmm_clean"
    n_fields = len(fields)
    os.makedirs(os.path.join(figure_savedir,folder_fields),exist_ok=True)
    
    full,rpe = load_data()
    if os.path.exists(cleaned_data_path):
        print("loading cleaned data")
        cleaned = pd.read_pickle(cleaned_data_path)
    else:
        
        cleaned = cleaning(full,rpe)
        cleaned.to_pickle(cleaned_data_path)


    if norm:
        if os.path.exists(normalized_data_path):
            print("loading normalized data")
            gps_data = pd.read_pickle(normalized_data_path)
        else:
            transformer = AthleteTransformer(data_fields=fields,col_time='tps',missing_value_threshold=0.5)
            gps_data = transformer.fit_transform(cleaned)
            gps_data.to_pickle(normalized_data_path)
    else:
        gps_data = cleaned

    template = "_{}comp_{}mix_{}iter_{}{}{}{}{}".format(n_comp,n_mix,n_iter,mat,
                                                        "_var" if var else "",
                                                        "_norm" if norm else "",
                                                        "_roll" if rolling else "",
                                                        "_prior" if prior else "")


    path_model = "model/hmm_"+folder_fields+template+".pkl"

    path_emission = os.path.join(figure_savedir,folder_fields,"emission/",template+".png")
    path_emission_box = os.path.join(figure_savedir,folder_fields,"emission_box/",template+".png")
    path_random = os.path.join(figure_savedir,folder_fields,"random/",template)
    path_time = os.path.join(figure_savedir,folder_fields,"time/",template+".png")
    # path_mean = os.path.join(figure_savedir,folder_fields,"mean/",template+".png")
    path_matrix = os.path.join(figure_savedir,folder_fields,"matrix/",template)
    path_score = os.path.join(figure_savedir,folder_fields,"score/",template+".txt")
    os.makedirs(os.path.join(figure_savedir,folder_fields,"emission"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"emission_box"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"random"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"time"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"mean"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"matrix"),exist_ok=True)
    os.makedirs(os.path.join(figure_savedir,folder_fields,"score"),exist_ok=True)


    # load the hmm model
    with open(path_model, 'rb') as handle:
        model = pickle.load(handle)
    
    #if the transition matrix is not close to id, we pass the model, we plot just the score
    if not np.allclose(model.transmat_,np.eye(n_comp),atol=5e-2):
        scores_model(model,gps_data[fields].values,gps_data.groupby('id').size().values,
                    path_score)
        return

    
    predicted = predict_states(gps_data,model,fields,label_states,var)
    
    generate_tikz_hmm(model,path_matrix,
                            label_states_ordered=label_states,
                            colors=palette)
    
    plot_times_in_state(predicted,rpe,
                        path_time,
                        label_states=label_states,
                        palette=palette)
    plot_random_sessions(predicted,
                         path_random,
                        label_states=label_states,
                        data_fields=fields,
                        palette=palette)

    if n_mix == 1:
        plot_emission_matrix(model,path_emission,
                            label_states=label_states,
                            data_fields=fields,
                            palette=palette)
        plot_gaussian_emission_boxplot(model,path_emission_box,
                            label_states=label_states,
                            data_fields=fields,
                            palette=palette)
        

    else:
        plot_gmm_emission_matrix(model,path_emission,
                            label_states=label_states,
                            data_fields=fields)
        plot_gmm_emission_boxplot(model,path_emission_box,
                            label_states=label_states,
                            data_fields=fields,
                            palette=palette)
    
    
    scores_model(model,gps_data[fields].values,gps_data.groupby('id').size().values,
                    path_score)
    

def extract_parameters(file):
    #format hmm_hr_3comp_1mix_500iter_diag_norm_roll_prior.pkl
    #or hmm_power_3comp_1mix_500iter_diag_norm_roll_prior.pkl
    # or hmm_hr_power_3comp_1mix_500iter_diag_norm_roll_prior.pkl
    fields = []
    if "hr" in file:
        fields.append("stream_heartrate")
    if "power" in file:
        fields.append("stream_watts")
    if "hr" in file and "power" in file:
        fields = ["stream_heartrate","stream_watts"]
        n_comp = int(file.split("_")[3].replace("comp",""))
        n_mix = int(file.split("_")[4].replace("mix",""))
        n_iter = int(file.split("_")[5].replace("iter",""))
        cov_type = file.split("_")[6].replace(".pkl","")
    else:
        n_comp = int(file.split("_")[2].replace("comp",""))
        n_mix = int(file.split("_")[3].replace("mix",""))
        n_iter = int(file.split("_")[4].replace("iter",""))
        cov_type = file.split("_")[5].replace(".pkl","")
    var = "var" in file
    norm = "norm" in file
    rolling = "roll" in file
    prior = "prior" in file
    return fields,n_comp,n_mix,n_iter,cov_type,var,norm,rolling,prior


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HMM plots')
    parser.add_argument('--nComp', type=int, default=3, help='Number of hidden states')
    parser.add_argument('--nMix', type=int, default=1, help='Number of mixture components')
    parser.add_argument('--nIter', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--covType', type=str, default='diag', help='Covariance type')
    parser.add_argument('-v,--variation', action='store_true', help='Use variation')
    parser.add_argument('-n,--normalized', action='store_true', help='Use normalized data')
    parser.add_argument('-r,--rolling', action='store_true', help='Use rolling data')
    parser.add_argument('--dataField', type=str, nargs='+', default=["stream_watts"], help='Fields to use for the training')
    parser.add_argument('-p,--prior', action='store_true', help='Use prior')
    args = parser.parse_args()
                        


    cleaned_data_path = "data/cleaned.pkl"
    normalized_data_path = "data/normalized.pkl"
    # var = False
    # norm = True
    # rolling = True
    # prior = True
    # mat = "diag"
    # n_comp = 3
    # n_mix = 1
    # n_iter = 1000
    # fields = ["stream_heartrate","stream_watts"]
    
    # var = args.variation
    # norm = args.normalized
    # rolling = args.rolling
    # mat = args.covType
    # n_comp = args.nComp
    # n_mix = args.nMix
    # n_iter = args.nIter
    # fields = args.dataField
    # prior = args.prior


    model_path = "model"

    os.listdir(model_path)
    for file in os.listdir(model_path):
        if "hmm" in file:
            fields,n_comp,n_mix,n_iter,mat,var,norm,rolling,prior = extract_parameters(file)
            if n_mix !=1 :
                continue
            plot_all_figures(n_comp,n_mix,n_iter,mat,var,norm,rolling,prior,fields)


   
    

  
    # plot the emission matrix
    # scores_model(model,training_df[args.dataField].values,training_df.groupby('id').size().values,score_template.format(args.nComp,args.nMix,args.nIter,args.covType,
    #                                                                                                                     "_var" if args.variation else "",
    #                                                                                                                     "_norm" if args.normalized else "",
    #                                                                                                                     "_roll" if args.rolling else ""))
    

    # scores_model(model,gps_data[fields].values,gps_data.groupby('id').size().values,
    #              score_template.format(n_comp,n_mix,n_iter,mat,
    #                                     "_var" if var else "",
    #                                     "_norm" if norm else "",
    #                                     "_roll" if rolling else ""))
    # predicted = predict_states(gps_data,model,fields,label_states,var)

    # plot_times_in_state(predicted,rpe,
    #                     time_template.format(n_comp,n_mix,
    #                                          n_iter,mat,
    #                                         "_var" if var else "",
    #                                         "_norm" if norm else "",
    #                                         "_roll" if rolling else ""),
    #                     label_states=label_states,
    #                     palette=palette)
    # plot_random_sessions(predicted,
    #                      random_template.format(n_comp,n_mix,
    #                                             n_iter,mat,
    #                                             "_var" if var else "",
    #                                             "_norm" if norm else "",
    #                                             "_roll" if rolling else ""),
    #                     label_states=label_states,
    #                     data_fields=fields,
    #                     palette=palette)

    # if n_mix == 1:
    #     plot_emission_matrix(model,emission_template.format(n_comp,n_mix,n_iter,mat,
    #                                                         "_var" if var else "",
    #                                                         "_norm" if norm else "",
    #                                                         "_roll" if rolling else ""),
    #                                                         label_states=label_states,
    #                                                         data_fields=fields,
    #                                                         palette=palette)
    #     plot_gaussian_emission_boxplot(model,emission_box_template.format(n_comp,n_mix,n_iter,mat,
    #                                                         "_var" if var else "",
    #                                                         "_norm" if norm else "",
    #                                                         "_roll" if rolling else ""),
    #                                                         label_states=label_states,
    #                                                         data_fields=fields,
    #                                                         palette=palette)
    # else:
    #     plot_gmm_emission_matrix(model,emission_template.format(n_comp,n_mix,n_iter,mat,
    #                                                         "_var" if var else "",
    #                                                         "_norm" if norm else "",
    #                                                         "_roll" if rolling else ""
    #                                                         ),label_states=label_states,
    #                                                         data_fields=fields
    #                                                         )
    #     plot_gmm_emission_boxplot(model,emission_box_template.format(n_comp,n_mix,n_iter,mat,
    #                                                         "_var" if var else "",
    #                                                         "_norm" if norm else "",
    #                                                         "_roll" if rolling else ""),
    #                                                         label_states=label_states,
    #                                                         data_fields=fields,
    #                                                         palette=palette)
        
    # generate_tikz_hmm(model,"hmm/"+matrix_template.format(n_comp,n_mix,n_iter,mat,
    #                                                         "_var" if var else "",
    #                                                         "_norm" if norm else "",
    #                                                         "_roll" if rolling else ""),
    #                                                         label_states=label_states,
    #                                                         colors=palette)

    
    # print("bug here, i don't know why so i wont appear")
    #     # plot_means_states(predicted,mean_template.format(args.nComp,args.nMix,args.nIter,args.covType,"var" if args.variation else "no_var","norm" if args.normalized else "no_norm"),label_states=label_states)
    

 
