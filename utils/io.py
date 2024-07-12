import pickle
import hmmlearn.hmm as hmm

def load_data():
    with open('df_clust_all.pkl', 'rb') as handle:
        full = pickle.load(handle)
        
    with open('df_rpe.pkl', 'rb') as handle:
        rpe = pickle.load(handle)
    return full,rpe


def write_model(model: hmm.BaseHMM, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)