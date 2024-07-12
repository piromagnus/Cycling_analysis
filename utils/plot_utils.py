from cProfile import label
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from regex import P
import seaborn as sns
import numpy as np
import hmmlearn.hmm as hmm
from subprocess import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import covariance




def get_times_in_state(cleaned,label_states=["low","medium","high"],id_key='id_session'):
    times_in_state = cleaned.groupby([id_key, 'state_label']).size().unstack().fillna(0)
  
    # normalize the time spent in each state
    times_in_state = times_in_state.div(times_in_state.sum(axis=1), axis=0)
    #reorder the columns to low, medium, high
    times_in_state = times_in_state[label_states]
    return times_in_state

def plot_graphs(gps:pd.DataFrame, ys=["stream_watts","stream_distance","stream_heartrate"]):
    fig, axs = plt.subplots(len(ys), 1,figsize=(10,5*len(ys)))
    for i,y in enumerate(ys):
        sns.lineplot(data=gps,x="tps",y=y,ax=axs[i],hue="id",palette="viridis")
        plt.title(y)
    plt.show()


def plot_times_in_state(predicted,meta_data,filename,label_states = ["low","medium","high"], palette = ['green','blue','red'],id_key='id_session'):
    
    label_states = predicted['state_label'].unique()
    times_in_state = get_times_in_state(predicted,label_states)
    #merge with date from rpe
    times_in_state = times_in_state.merge(meta_data[[id_key,'date','rpe']], left_index=True, right_on=id_key)
    times_in_state.set_index('id_session', inplace=True)
    #TODO : normalize the rpe with min and max
    times_in_state.loc[times_in_state['rpe']==0,'rpe']=np.nan
    times_in_state=times_in_state.dropna(subset=['rpe'])
    times_in_state['rpe'] = (times_in_state['rpe'] - times_in_state['rpe'].min()) / (times_in_state['rpe'].max() - times_in_state['rpe'].min())

    # group by month and mean the time spent in each state
    times_in_state['date'] = pd.to_datetime(times_in_state['date'])
    times_in_state['date'] = times_in_state['date'].dt.to_period('M')
    times_in_state = times_in_state.groupby('date').mean()
    times_in_state.reset_index(inplace=True)
    times_in_state['date'] = times_in_state['date'].dt.to_timestamp()
    times_in_state.set_index('date', inplace=True)

    # plot the time spent in each state for each session label by date (dt_session)
    plt.stackplot(times_in_state.index, times_in_state[label_states].T, labels=label_states, colors=palette)
    plt.plot(times_in_state.index, times_in_state['rpe'], label='RPE', color='black', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Time spent in each state')
    plt.title('Time spent in each state by month')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    # plt.show()



def plot_emission_matrix(model,filename,label_states = ["low","medium","high"],data_fields=["stream_watts"],palette=['green','blue','red']):
    #TODO : add the possibility of matrix which is not diagonal
    nb_fields=len(data_fields)
    means = model.means_.reshape(-1,nb_fields)
    covariances =np.array([np.diag(cov) for cov in model.covars_]).reshape(-1,nb_fields)
    # print(covariances)
    order = np.argsort(means,axis=0)[:,-1]
    # print(order)
    means = means[order]
    covariances = covariances[order]
    # print(covariances,means)


    nb_components = means.shape[0]
    # print(nb_components)
    for field in range(nb_fields):
        plt.figure(figsize=(10, 5))
        for state in range(nb_components):
            
            
            mean = means[state,field]
            variance = covariances[state,field]

            # Assuming a 1D Gaussian for simplicity, adjust if your data is multi-dimensional
            x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 1000).reshape(-1, 1)
            y = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * (x - mean)**2 / variance).reshape(-1, 1)

            plt.plot(x, y, label=label_states[state], color=palette[state])

        plt.xlabel('Power' if data_fields[field] == 'stream_watts' else 'Heartrate' if data_fields[field] == 'stream_heartrate' else 'Speed')
        plt.ylabel('Density')
        plt.title('Gaussian Distribution of Power for each State')
        plt.legend()
        plt.savefig(filename.replace(".png","_{}.png".format(data_fields[field])))
        plt.close()
    # plt.show()

def plot_gmm_emission_matrix(model: hmm.GMMHMM,
                             filename, 
                             palette = ['green','blue','red'],
                             label_states=["low","medium","high"],
                             data_fields=["stream_watts"]):
    
    
    #TODO : modify pour les différents champs

    nb_fields=len(data_fields)
    n_states = model.n_components
    n_mix = model.n_mix
    # deal with the mixture components and the data fields
    # means is of shape (n_states,n_mix,nb_fields)
    # covariances is a matrix of size (n_states, n_mix, nb_fields) because it is diagonal

    #model is a GMMHMM object
    covariances = model.covars_
    means = model.means_
    # print(means)
    # print(covariances.shape)
    # sort by states for all the fields and all the mixtures
    order = np.argsort(means.reshape(n_states,-1),axis=0)
    # keep only 1 dimension but take in the majority position of the fields and mixtures for each state
    order = order.mean(axis=1).round().astype(int)
    # print(order)
    means = means[order]
    covariances = covariances[order]
    # print(means)
    # print(covariances)
    states = label_states
    # box plot of the means with variance for each state
    # means contains the means of each mixture component for each state
    # covariances contains the variances of each mixture component for each state
    for field in range(nb_fields):
        plt.figure(figsize=(10, 5))
        for state in range(n_states):
            for mixture in range(n_mix):
                mean = means[state][mixture][field]
                variance = covariances[state][mixture][field]
                # print(mean,variance)

                # Assuming a 1D Gaussian for simplicity, adjust if your data is multi-dimensional
                x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 1000).reshape(-1)
                y = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * (x - mean)**2 / variance).reshape(-1)

                plt.plot(x, y, label=states[state] + ' Mixture ' + str(mixture), color=palette[state],alpha=0.5)
                #fill the area under the curve
                plt.fill_between(x, y, color=palette[state], alpha=0.5)
        plt.xlabel('Power' if data_fields[field] == 'stream_watts' else 'Heartrate' if data_fields[field] == 'stream_heartrate' else 'Speed')
        plt.ylabel('Density')
        plt.title('Gaussian Mixture Model of Power for each State')
        plt.legend()
        plt.savefig(filename.replace(".png","_{}.png".format(data_fields[field])))
        plt.close()

def plot_gmm_emission_boxplot(model: hmm.GMMHMM,
                                filename, 
                                palette = ['green','blue','red'],
                                label_states=["low","medium","high"],
                                data_fields=["stream_watts"]):
        
        nb_fields=len(data_fields)
        n_states = model.n_components
        n_mix = model.n_mix
        # deal with the mixture components and the data fields
        # means is of shape (n_states,n_mix,nb_fields)
        # covariances is a matrix of size (n_states, n_mix, nb_fields) because it is diagonal
    
        #model is a GMMHMM object
        covariances = model.covars_
        means = model.means_
        # print(means)
        # print(covariances.shape)
        # sort by states for all the fields and all the mixtures
        order = np.argsort(means.reshape(n_states,-1),axis=0)
        # keep only 1 dimension but take in the majority position of the fields and mixtures for each state
        order = order.mean(axis=1).round().astype(int)
        # print(order)
        means = means[order]
        covariances = covariances[order]
        states = label_states
        # box plot of the means with variance for each state
        # means contains the means of each mixture component for each state
        # covariances contains the variances of each mixture component for each state
        for field in range(nb_fields):
            fig, axs = plt.subplots(1, n_states, figsize=(10, 5))
            # plot 2 box plots for each state
            for state in range(n_states):
                for mixture in range(n_mix):
                    axs[state].boxplot(np.random.normal(means[state][mixture][field], np.sqrt(covariances[state][mixture][field]), 1000),
                        positions=[mixture],
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor='none', color=palette[state]),  # Box color
                        whiskerprops=dict(color=palette[state]),  # Whisker color
                        capprops=dict(color=palette[state]),  # Cap color
                        medianprops=dict(color=palette[state]),  # Median color
                        meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))
                axs[state].set_ylim(-200, 1000) if data_fields[field] == 'stream_watts' else axs[state].set_ylim(-50, 250) if data_fields[field] == 'stream_heartrate' else axs[state].set_ylim(0, 20)
                axs[state].set_title(states[state])
                axs[state].set_xlabel('Mixture')
            axs[0].set_ylabel('Power' if data_fields[field] == 'stream_watts' else 'Heartrate' if data_fields[field] == 'stream_heartrate' else 'Speed')
            plt.suptitle('Gaussian Mixture Model of Power for each State')
            plt.savefig(filename.replace(".png","_{}.png".format(data_fields[field])))
            plt.close()


def plot_gaussian_emission_boxplot(model,filename="gaussian_emission_boxplot_3_var.png",
                                   label_states=["low","medium","high"],
                                   data_fields=["stream_watts"],
                                   palette=['green','blue','red']):
    nb_fields=len(data_fields)
    means = model.means_.reshape(-1,nb_fields)
    covariances =np.array([np.diag(cov) for cov in model.covars_]).reshape(-1,nb_fields)
    order = np.argsort(means,axis=0)[:,-1]
    means = means[order]
    covariances = covariances[order]
    nb_components = len(means)
    # print(means.shape)
    # box plot of the means with variance as the whiskers for each state
    # means contains the means for each state
    # covariances contains the variances for each state
    for field in range(nb_fields):
        # print(means,covariances)
        means_ = means[:,field]
        covariances_ = covariances[:,field]
        # print(means_,covariances_)
        plt.figure(figsize=(10, 5))
        # plt.boxplot([np.random.normal(mean, np.sqrt(variance), 1000) for mean, variance in zip(means_, covariances_)], positions=range(nb_components),
        #             patch_artist=True, showfliers=False, boxprops=dict(facecolor='none', color=palette), whiskerprops=dict(color=palette),
        #             capprops=dict(color=palette), medianprops=dict(color=palette), meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))
        for i, (mean, variance) in enumerate(zip(means_, covariances_)):
            plt.boxplot(np.random.normal(mean, np.sqrt(variance), 1000),
                positions=[i],
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='none', color=palette[i]),  # Box color
                whiskerprops=dict(color=palette[i]),  # Whisker color
                capprops=dict(color=palette[i]),  # Cap color
                medianprops=dict(color=palette[i]),  # Median color
                meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))  # Mean color
        plt.xticks(range(nb_components), label_states)
        plt.xlabel('State')
        plt.ylabel('Power' if data_fields[field] == 'stream_watts' else 'Heartrate' if data_fields[field] == 'stream_heartrate' else 'Speed')
        plt.title('Gaussian Distribution for each State')
        plt.savefig( filename.replace(".png","_{}.png".format(data_fields[field])))
        plt.close()
    # plt.show()

def plot_gaussian_full_emission_boxplot(model,filename="gaussian_full_emission_boxplot_3_var.png",
                                      label_states=["low","medium","high"],
                                        data_fields=["stream_watts","stream_heartrate"],
                                        palette=['green','blue','red']):
    
    nb_fields=len(data_fields)

    means = model.means_
    covariances = model.covars_
    nb_components = len(means)

    for state in range(nb_components):
        mean= means[state]
        cov = covariances[state]
        plot_2d_gaussian_subplots(mean, cov, filename=filename.replace(".png","_{}.png".format(label_states[state])),color=palette[state],state_label=label_states[state],data_field=data_fields)


def plot_2d_gaussian_subplots(distributions, filename=None, ellipse_colors=['b', 'g', 'r'], labels=['X-axis', 'Y-axis']):
    """
    Plots subplots of 2D representations of bivariate Gaussian distributions with surfaces for the mean and
    elliptic lines for the standard deviations.
    
    Parameters:
    - distributions: list of tuples
        Each tuple contains (mean, cov) for the Gaussian distribution.
    - filename: str, optional
        The name of the file to save the plot. If None, the plot is not saved.
    - ellipse_colors: list of str, optional
        The colors of the ellipses representing the covariances.
    - labels: list of str, optional
        The labels of the variables (X and Y axes).
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    all_x = []
    for i, (mean, cov) in enumerate(distributions):
        ax = axs[i]

        # Plot the mean
        ax.plot(mean[0], mean[1], 'ro')  # Mean point in red

        # Generate ellipses for -1 and +1 std, and a filled ellipse for the mean
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        
        # The angle of the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Width and height of the ellipses for std deviations
        width_2std, height_2std = 2 * np.sqrt(eigenvalues)
        width_1std, height_1std = np.sqrt(eigenvalues)

        # Ellipse at 2 standard deviations
        ell_2std = Ellipse(xy=mean, width=width_2std, height=height_2std, angle=angle, edgecolor=ellipse_colors[i], fc='None', lw=2, linestyle='--')
        ax.add_patch(ell_2std)

        # Ellipse at 1 standard deviation
        ell_1std = Ellipse(xy=mean, width=width_1std, height=height_1std, angle=angle, edgecolor=ellipse_colors[i], fc='None', lw=2, linestyle=':')
        ax.add_patch(ell_1std)

        # Filled ellipse for the mean
        filled_ellipse = Ellipse(xy=mean, width=width_1std, height=height_1std, angle=angle, edgecolor=ellipse_colors[i], fc=ellipse_colors[i], alpha=0.2)
        ax.add_patch(filled_ellipse)

        all_x.extend([mean[0] - 3*np.sqrt(cov[0, 0]), mean[0] + 3*np.sqrt(cov[0, 0])])
        ax.set_ylim(mean[1] - 3*np.sqrt(cov[1, 1]), mean[1] + 3*np.sqrt(cov[1, 1]))

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(f'Distribution {i+1}')
        ax.grid(True)

    x_limits = [min(all_x), max(all_x)]
    for ax in axs:
        ax.set_xlim(x_limits)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
def plot_3d_gaussian_subplots(distributions, filename=None, labels=['X-axis', 'Y-axis'],state_label=["low","medium","high"]):
    """
    Plots subplots of 3D representations of bivariate Gaussian distributions.
    
    Parameters:
    - distributions: list of tuples
        Each tuple contains (mean, cov) for the Gaussian distribution.
    - filename: str, optional
        The name of the file to save the plot. If None, the plot is not saved.
    - labels: list of str, optional
        The labels of the variables (X and Y axes).
    """
    fig = plt.figure(figsize=(15, 5))

    all_x = []
    for i, (mean, cov) in enumerate(distributions):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        x, y = np.mgrid[mean[0] - 3*np.sqrt(cov[0, 0]):mean[0] + 3*np.sqrt(cov[0, 0]):.1, 
                        mean[1] - 3*np.sqrt(cov[1, 1]):mean[1] + 3*np.sqrt(cov[1, 1]):.1]
        pos = np.dstack((x, y))
        rv = multivariate_normal(mean, cov)
        z = rv.pdf(pos)

        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

        all_x.extend([mean[0] - 3*np.sqrt(cov[0, 0]), mean[0] + 3*np.sqrt(cov[0, 0])])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel('Probability Density')
        ax.set_title(f'State {state_label[i]}')

    x_limits = [min(all_x), max(all_x)]
    for ax in fig.axes:
        ax.set_xlim(x_limits)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_3d_gaussian(mean, cov, filename=None, labels=['X-axis', 'Y-axis'],state_label="low"):
    """
    Plots a 3D representation of a bivariate Gaussian distribution.
    
    Parameters:
    - mean: array-like, shape (2,)
        The mean of the Gaussian distribution.
    - cov: array-like, shape (2, 2)
        The covariance matrix of the Gaussian distribution.
    """
    x, y = np.mgrid[mean[0] - 3*np.sqrt(cov[0, 0]):mean[0] + 3*np.sqrt(cov[0, 0]):.1, 
                    mean[1] - 3*np.sqrt(cov[1, 1]):mean[1] + 3*np.sqrt(cov[1, 1]):.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Gaussian Distribution')
    plt.show()

def plot_2d_heatmap_subplots(distributions, filename=None, labels=['X-axis', 'Y-axis'],state_label=["low","medium","high"]):
    """
    Plots subplots of 2D heatmap representations of bivariate Gaussian distributions.
    
    Parameters:
    - distributions: list of tuples
        Each tuple contains (mean, cov) for the Gaussian distribution.
    - filename: str, optional
        The name of the file to save the plot. If None, the plot is not saved.
    - labels: list of str, optional
        The labels of the variables (X and Y axes).
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    all_x = []
    for i, (mean, cov) in enumerate(distributions):
        ax = axs[i]

        x, y = np.mgrid[mean[0] - 3*np.sqrt(cov[0, 0]):mean[0] + 3*np.sqrt(cov[0, 0]):.1, 
                        mean[1] - 3*np.sqrt(cov[1, 1]):mean[1] + 3*np.sqrt(cov[1, 1]):.1]
        pos = np.dstack((x, y))
        rv = multivariate_normal(mean, cov)
        z = rv.pdf(pos)

        ax.contourf(x, y, z, cmap='viridis')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(f'State {state_label[i]}')

        all_x.extend([mean[0] - 3*np.sqrt(cov[0, 0]), mean[0] + 3*np.sqrt(cov[0, 0])])

    x_limits = [min(all_x), max(all_x)]
    for ax in axs:
        ax.set_xlim(x_limits)

    plt.colorbar(axs[0].collections[0], ax=axs, orientation='horizontal', fraction=.1)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_2d_heatmap(mean, cov):
    """
    Plots a 2D heatmap representation of a bivariate Gaussian distribution.
    
    Parameters:
    - mean: array-like, shape (2,)
        The mean of the Gaussian distribution.
    - cov: array-like, shape (2, 2)
        The covariance matrix of the Gaussian distribution.
    """
    x, y = np.mgrid[mean[0] - 3*np.sqrt(cov[0, 0]):mean[0] + 3*np.sqrt(cov[0, 0]):.1, 
                    mean[1] - 3*np.sqrt(cov[1, 1]):mean[1] + 3*np.sqrt(cov[1, 1]):.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)

    plt.figure()
    plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Gaussian Distribution Heatmap')
    plt.show()
    

def plot_random_sessions(cleaned:pd.DataFrame,
                         filename:str,
                         label_states=["low","medium","high"],
                         palette=['green','blue','red'],
                         data_fields=["stream_watts"],
                         id_key='id_session'):
    ids = cleaned[id_key].unique()
    np.random.seed(0)
    sample_id = np.random.choice(ids,10)
    for i in sample_id:
        sample_session = cleaned[cleaned[id_key] == i]

        for field in data_fields:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(data=sample_session,x='tps',y=field,hue='state_label',hue_order=label_states,palette=palette[:len(label_states)])
            plt.ylabel('Power' if field == 'stream_watts' else 'Heartrate' if field == 'stream_heartrate' else 'Speed')
            plt.title('Random Sessions for each State for id {}'.format(i))
            os.makedirs( filename, exist_ok=True)
            plt.savefig(os.path.join(filename,field+"_session_{}_.png".format(i)))
            plt.legend()
            plt.close()
        # plt.show()
    
def plot_means_states(predicted:pd.DataFrame,filename:str,palette=["green","blue","red"],label_states=["low","medium","high"]):
    new_cleaned = predicted.copy()  
    new_cleaned.set_index('id',inplace=True)
    # print(new_cleaned.isnull().groupby('id').mean().sum())
    new_cleaned['dt_session'] = pd.to_datetime(new_cleaned['dt_session'])
    new_cleaned['dt_session'] = new_cleaned['dt_session'].dt.to_period('M')

    power_month = new_cleaned[['dt_session','state_label','stream_watts']].groupby(['dt_session','state_label']).mean()['stream_watts'].reset_index()
    power_month.set_index('dt_session',inplace=True)

    # print(power_month)
    # power_month = power_month.pivot(columns='state_label',values='stream_watts')

    plt.figure(figsize=(10,5))
    sns.lineplot(data=power_month,x=power_month.index,y='stream_watts',hue='state_label',hue_order=label_states,palette=palette)
    plt.xlabel('Month')
    plt.ylabel('Power')
    plt.title('Mean Power for each State in each Month')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_matrix_transition(model,filename,labels=['low','medium','high'],colors=['green','blue','red']):

    reduc_mat = model.transmat_
    print(reduc_mat)
    fig, ax = plt.subplots(figsize=(20,20))
    graph = TransGraph.from_array(reduc_mat, labels=labels)
    colors = {labels[i]: f'C{i}' for i in range(len(labels))}

    plt.figure(figsize=(6, 6))

    graph.draw(ax,edgecolors=colors, edgelabels=True, edgewidths=2)
    plt.title('Transition Matrix')
    plt.legend()
    plt.savefig(os.path.join('hmm', filename))
    plt.close()

def generate_tikz_hmm(model, filename,label_states, colors):
    """
    Generate a TikZ diagram of the HMM transition matrix.

    Parameters:
    model (hmmlearn model): The HMM model.
    labels (list): The labels for each state.
    colors (list): The colors for each state.
    filename (str): The output filename for the LaTeX file.
    Returns:
    None
    """
    trans_matrix = model.transmat_
    order = model.means_.argsort(axis=0)[:, 0]
    print(order)
    print(trans_matrix)
    # label_states_ordered = [label_states_ordered[i] for i in order]
    # print(label_states_ordered)
    # colors = [colors[i] for i in order]
    #reorder the transition matrix
    n_components = len(label_states)
    
    # Header for the LaTeX file
    tikz_header = r"""
    \documentclass{standalone}
    \usepackage{tikz}
    \usetikzlibrary{arrows, automata, positioning}
    \begin{document}
    \begin{tikzpicture}[shorten >=1pt, node distance=4cm, on grid, auto]
    """

    #position of the nodes circular layout
    positions = [(5*np.cos(2*np.pi/n_components*i), 5*np.sin(2*np.pi/n_components*i)) for i in range(n_components)]

    # Define nodes

    tikz_nodes = ""
    for i,order in enumerate(order):
        x,y = positions[order]
        tikz_nodes += f"\\node[state, fill={colors[i]}] (S{order}) at ({int(x)},{int(y)}) {{${label_states[i]}$}} ;\n" # Add closing curly brace at the end

    # Define edges
    tikz_edges = ""
    for i in range(n_components):
        for j in range(n_components):
            if trans_matrix[i, j] > 1e-6:
                if i == j:
                    # Self-loop
                    tikz_edges += f"\\path[->] (S{i}) edge [loop above] node {{\\({trans_matrix[i, j]:.2f}\\)}} (S{i});\n"
                else:
                    # Edge to another state
                    tikz_edges += f"\\path[->] (S{i}) edge [bend left] node {{\\({trans_matrix[i, j]:.2f}\\)}} (S{j});\n"
                    tikz_edges += f"\\path[->] (S{j}) edge [bend left] node {{\\({trans_matrix[j, i]:.2f}\\)}} (S{i});\n"

    # Footer for the LaTeX file
    tikz_footer = r"""
    \end{tikzpicture}
    \end{document}
    """

    # Combine all parts
    tikz_content = tikz_header + tikz_nodes + tikz_edges + tikz_footer

    # Write to file
    f= open(filename+".tex","w")
    f.write(tikz_content)
    f.close()

    try:
        # output to the same filename with pdf
        dir = os.path.dirname(filename)
        output_name = os.path.basename(filename)

        # run(['pdflatex', '-output-directory='+dir, "-jobname="+output_name, filename+".tex"])
        try:
            result = run(
            ['pdflatex', '-output-directory=' + dir, '-jobname=' + output_name, filename + '.tex'],
            check=True, 
            capture_output=True, 
            text=True
        )
        except CalledProcessError as e:
            print("An error occurred while running pdflatex:")
            print(e.stderr)
        else:
            print("pdflatex ran successfully.")

        # cleanup
        os.remove(filename+".aux")
        os.remove(filename+".log")

    except CalledProcessError as e:
        print(f"Error occurred during LaTeX compilation: {e}")
   
def scores_model(model:hmm.BaseHMM,X:np.array,lengths:list,filename:str):
    aic = model.aic(X,lengths)
    bic = model.bic(X,lengths)
    score = model.score(X,lengths)
    with open(filename,'w') as file:
        file.write("AIC : {}\n".format(aic))
        file.write("BIC : {}\n".format(bic))
        file.write("Score : {}\n".format(score))