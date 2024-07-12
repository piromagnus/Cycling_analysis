from curses import meta
import time
from cv2 import norm
import pandas as pd
import numpy as np
from .normalizer import AthleteTransformer
from tqdm import tqdm


def session_id_to_ath_id(session_id:int) : return int(session_id/1000000)

def remove_long_nan_sequences_v2(df, column_name, th, id_key="id"):
    # Create a mask of NaN values
    nan_mask = df[column_name].isna()
    
    # Create a cumulative sum to identify sequences
    df['group'] = nan_mask.groupby(df[id_key]).cumsum()
    df['sequence_id'] = df['group'] * nan_mask

    # Compute the lengths of each sequence
    sequence_lengths = df.groupby([id_key, 'sequence_id']).size()
    long_sequences = sequence_lengths[sequence_lengths > th].index

    # Filter out the long sequences
    mask = ~df.set_index([id_key, 'sequence_id']).index.isin(long_sequences)
    
    # Drop the temporary columns used for computation
    df = df.drop(columns=['group', 'sequence_id'])

    # Return the filtered DataFrame
    return df[mask]

def remove_long_nan_sequences_v3(df, column_name, th, id_key='id'):
    def filter_group(group):
        mask = group[column_name].isna()
        # Label consecutive NaNs uniquely
        seq_id = (mask != mask.shift()).cumsum()
        # Filter out long NaN sequences
        sequence_lengths = mask.groupby(seq_id).transform('sum')
        keep_mask = ~mask | (sequence_lengths <= th)
        return group[keep_mask]

    return df.groupby(id_key, group_keys=False).apply(filter_group)

def new_cleaning(gps:pd.DataFrame,rpe=None,th_long=300,th_outliers=3,small_session=200,hr_min=20,watt_min=10,id_key="id",fields=["stream_watts","stream_heartrate"]):

    # TODO : 
    # - remove the with constant distance
    #turn all the negative values to nan
    for field in fields+['tps']:
        gps.loc[gps[field]<0,field] = np.nan

    print("shape before cleaning : ",gps.shape)
    print('nb of sessions : ',gps[id_key].nunique())
    #remove the long sequences of nan values in watts and hr
    for field in fields:
        a=time.time()
        gps = remove_long_nan_sequences_v3(gps, field, th_long,id_key=id_key)
        print("time to remove long nan sequences : ",time.time()-a)
    print("shape after removing the long sequences of nan values : ",gps.shape)
    print('nb of sessions : ',gps[id_key].nunique())
    
    cleaned = gps.copy()
    if cleaned.index.duplicated().any():
        raise ValueError("Duplicate indices found in the DataFrame")
    isnull = cleaned.isnull().groupby(id_key).sum()
    print("number of missing values group by id : ",isnull.sum())
    for field in fields:
        cleaned = interpolate_missing_values(cleaned, field,id_key=id_key)
    
    print("shape after interpolation : ",cleaned.shape)
    print('nb of sessions : ',cleaned[id_key].nunique())

    isnull = cleaned.isnull().groupby(id_key).sum()
    print("number of missing values group by id after interpolation : ",isnull.sum())
    print('nb of sessions : ',cleaned[id_key].nunique())

    # remove the outliers
    for field in fields:
        cleaned = filter_outliers(cleaned, field, th_outliers)
    print("shape after removing the outliers : ",cleaned.shape)
    print('nb of sessions : ',cleaned[id_key].nunique())

    #remove the sessions with less than 200 frames
    count = cleaned.groupby(id_key).count()["tps"]
    cleaned = cleaned[cleaned["id"].isin(count[count>small_session].index)]
    print("shape after removing the sessions with less than {} frames: ".format(small_session),cleaned.shape)
    print('nb of sessions : ',cleaned[id_key].nunique())

    # remove the sessions with less than 20 hr
    mean_hr = cleaned[['stream_heartrate',id_key]].groupby(id_key).mean()["stream_heartrate"]
    cleaned = cleaned[cleaned[id_key].isin(mean_hr[mean_hr>hr_min].index)]
    print("shape after removing the id with less than {} hr : ".format(hr_min),cleaned.shape)
    print('nb of sessions : ',cleaned[id_key].nunique())

    # remove the sessions with less than 10 watts
    mean_watts = cleaned[['stream_watts',id_key]].groupby(id_key).mean()["stream_watts"]
    cleaned = cleaned[cleaned[id_key].isin(mean_watts[mean_watts>watt_min].index)]
    print("shape after removing the id with less than {} watts : ".format(watt_min),cleaned.shape)
    print('nb of sessions : ',cleaned[id_key].nunique())

    dates = rpe[["id_session","dt_session"]].drop_duplicates()
    cleaned = cleaned.merge(dates, how="inner", left_on=id_key, right_on="id_session")
    # print(cleaned.head())
    cleaned.drop(columns=["id_session"],inplace=True)
    cleaned.sort_values(by=['dt_session',"tps"],inplace=True)

    #reindex the tps with the size of the sequences
    cleaned["tps"] = cleaned.groupby(id_key).cumcount()

    #rename the id column to id_session
    cleaned.rename(columns={id_key:"id_session"},inplace=True)
    #rename the dt_session column to date
    cleaned.rename(columns={"dt_session":"date"},inplace=True)
    return cleaned[["id_session","tps","stream_watts","stream_heartrate","date"]]


def get_norm_data(cleaned:pd.DataFrame,meta_data:pd.DataFrame,period=60,max_ppr=1000,id_key="id"):
    norm_data = pd.DataFrame()
    ids = cleaned[id_key].unique()
    meta_selected = meta_data[meta_data[id_key].isin(ids)].copy()
    meta_selected.sort_values(by=['date'],inplace=True)
    meta_selected.loc[:,"order_id"]=np.arange(meta_selected.shape[0])
    # print(meta_selected.head())
    merged=cleaned.merge(meta_selected[[id_key,"order_id"]], how="inner", left_on=id_key, right_on=id_key)
    # print(merged.head())
    merged.sort_values(by=['order_id',"tps"],inplace=True)


    #TODO check the order after the groupby
    if "stream_heartrate" in cleaned.columns:
        ma_hr= merged.groupby('order_id')["stream_heartrate"].mean().rolling(period,min_periods=1).mean().reset_index()
        roll_std_hr = merged.groupby('order_id')["stream_heartrate"].mean().rolling(period,min_periods=1).std().reset_index()
        # max_hr = merged.groupby('order_id')["stream_heartrate"].max().rolling(period,min_periods=1).mean().reset_index()

        mean_std= roll_std_hr["stream_heartrate"].mean()
        roll_std_hr = roll_std_hr.fillna(mean_std)
        ma_hr=ma_hr.ffill().bfill()
        roll_std_hr=roll_std_hr.ffill().bfill()

        roll_max_hr = merged.groupby('order_id')["stream_heartrate"].max().rolling(period,min_periods=1).mean().reset_index()
        #print all the values
        # print("ma_hr : ",ma_hr["stream_heartrate"].values)
        # print("roll_std_hr : ",roll_std_hr["stream_heartrate"].values)
        # print("roll_max_hr : ",roll_max_hr["stream_heartrate"].values)
        # print("rpe : ",meta_data[["rpe","date","sport"]].values)
        norm_data = pd.DataFrame(index=ma_hr['order_id'].values, columns=['ma_hr','roll_std_hr','roll_max_hr'], 
                             data=np.column_stack([ma_hr["stream_heartrate"].values, roll_std_hr["stream_heartrate"].values,
                                                    roll_max_hr["stream_heartrate"].values]))
        

    if "stream_watts" in cleaned.columns:
        ppr = merged.groupby('order_id', group_keys=False)["stream_watts"].rolling(30,min_periods=1).mean().reset_index()##PPR30
        ppr = ppr.groupby('order_id')["stream_watts"].max().rolling(period,min_periods=1).max().reset_index()#Max on a period of time
        ppr[ppr > max_ppr]["stream_watts"] = max_ppr
        #smooth the ppr
        ppr.loc[:,"stream_watts"] = ppr["stream_watts"].rolling(30,min_periods=1).mean().reset_index()
        ppr = ppr.ffill().bfill()
        if norm_data.empty:
            norm_data = pd.DataFrame(index=ppr['order_id'].values, columns=['ppr'], 
                             data=np.column_stack([ppr["stream_watts"].values]))
        else:
            norm_data = pd.DataFrame(index=ppr['order_id'].values, columns=['ppr','ma_hr','roll_std_hr','roll_max_hr'], 
                             data=np.column_stack([ppr["stream_watts"].values, 
                                                   norm_data["ma_hr"].values, norm_data["roll_std_hr"].values,
                                                     norm_data["roll_max_hr"].values]))
    norm_data = norm_data.merge(meta_selected[[id_key,'order_id',"rpe","date","sport"]], how="inner", left_index=True, right_on='order_id')
    norm_data.set_index(id_key,inplace=True)

    return norm_data


def new_normalize(cleaned:pd.DataFrame,
                  meta_data:pd.DataFrame,
                  fields=["stream_watts"],
                  period=60,
                  rolling=False,
                  max_ppr=1000,
                  method_hr="mean",
                  id_key="id",
                  keep_cols=[]):
    norm_data = get_norm_data(cleaned,meta_data,period=period,max_ppr=max_ppr,id_key=id_key)
    
    # print(norm_data.head())

    normalizer = AthleteTransformer(data_fields=fields,col_time="tps",id_col=id_key,missing_value_threshold=0.5,
                                    rolling=rolling,method_hr=method_hr,
                                    norm_data=norm_data,
                                    keep_cols=keep_cols)
    normalized_df = normalizer.fit_transform(cleaned)
    if keep_cols != []:
        for col in keep_cols:
            col_name = col+"_x"
            if col_name in normalized_df.columns:
                normalized_df.drop(columns=[col_name],inplace=True)
                normalized_df.rename(columns={col+"_y":col},inplace=True)
                normalized_df.reset_index(inplace=True)
    
    # print(normalized_df.head())

    return normalized_df[keep_cols+fields+[id_key,"tps"]],norm_data

def normalize_data(cleaned:pd.DataFrame,rpe=None,fields=["stream_watts"],period=60,rolling=False,max_ppr=1000,method_hr="mean"):

    print(cleaned.head())
    ma_hr= cleaned.groupby("id")["stream_heartrate"].mean().rolling(period,min_periods=1).mean().reset_index()
    roll_std_hr = cleaned.groupby("id")["stream_heartrate"].mean().rolling(period,min_periods=1).std().reset_index()
    mean_std= roll_std_hr["stream_heartrate"].mean()
    roll_std_hr = roll_std_hr.fillna(mean_std)
    roll_max_hr = cleaned.groupby("id")["stream_heartrate"].mean().rolling(period,min_periods=1).max().reset_index()
    ppr = cleaned.groupby('id', group_keys=False)["stream_watts"].rolling(30,min_periods=1).mean().reset_index()##PPR30
    ppr = ppr.groupby('id')["stream_watts"].max().rolling(period,min_periods=1).max().reset_index()#Max on a period of time
    ppr[ppr > max_ppr]["stream_watts"] = max_ppr
    #smooth the ppr
    ppr.loc[:,"stream_watts"] = ppr["stream_watts"].rolling(30,min_periods=1).mean().reset_index()

    # index = cleaned.index
    # ma_hr = pd.merge(cleaned[['id']], ma_hr, on='id', how='right')
    # ma_hr = ma_hr.set_index(index)
    # roll_std_hr = pd.merge(cleaned[['id']], roll_std_hr, on='id', how='right')
    # roll_std_hr = roll_std_hr.set_index(index)
    # roll_max_hr = pd.merge(cleaned[['id']], roll_max_hr, on='id', how='right')
    # roll_max_hr = roll_max_hr.set_index(index)
    # cleaned.loc[:,"ma_hr"] = ma_hr["stream_heartrate"].values
    # cleaned.loc[:,"roll_std_hr"] = roll_std_hr["stream_heartrate"].values
    # cleaned.loc[:,"roll_max_hr"] = roll_max_hr["stream_heartrate"].values

    
    
    # # print('PPR: {}'.format(self.ppr))
    # # get a dataframe of size X with the self.ppr values identical for each id
    # ppr = pd.merge(cleaned[['id']], ppr, on='id', how='right')
    # ppr = ppr.set_index(index)

    # gather all the data in a single dataframe with the id as index
    norm_data = pd.DataFrame(index=ppr["id"].values, columns=['ppr','ma_hr','roll_std_hr','roll_max_hr'], 
                             data=np.column_stack([ppr["stream_watts"].values, 
                                                   ma_hr["stream_heartrate"].values, roll_std_hr["stream_heartrate"].values,
                                                     roll_max_hr["stream_heartrate"].values,
                                                     ]))
    normalizer = AthleteTransformer(data_fields=fields,col_time="tps",missing_value_threshold=0.5,
                                    rolling=rolling,method_hr=method_hr,norm_data=norm_data)
    normalized_df = normalizer.fit_transform(cleaned)
    if rpe is not None:
        normalized_df = normalized_df.merge(rpe, how="inner", left_on="id", right_on="id_session")
    # normalized_df.set_index(cleaned.index,inplace=True)

    #add the ppr, ma_hr, roll_std_hr, roll_max_hr
 

    return normalized_df[["id","tps","stream_watts","stream_heartrate"]]


# Function to remove NaN sequences longer than threshold
def remove_long_nan_sequences(df,column_name,th,id_key="id"):
    indices_to_keep = []

    def filter_nan_sequences(group):
        mask = group[column_name].isna()
        start = 0
        count = 0
        group_indices_to_keep = []

        for i, is_nan in enumerate(mask):
            if is_nan:
                count += 1
            else:
                if count <= th:
                    group_indices_to_keep.extend(group.index[start:start + count])
                group_indices_to_keep.append(group.index[i])
                start = i + 1
                count = 0

        if count <= th:  # Check at the end of the group
            group_indices_to_keep.extend(group.index[start:start + count])

        indices_to_keep.extend(group_indices_to_keep)
    
    df.groupby(id_key, group_keys=False).apply(filter_nan_sequences, include_groups=False)
    return df.loc[indices_to_keep]


def interpolate_missing_values(df, column,id_key="id"):
    # Interpolate missing values
    df[column] = df.groupby(id_key)[column].transform(lambda x: x.interpolate())
    
    # Fill remaining missing values with the next value
    df[column] = df[column].bfill().ffill()
    
    return df

def filter_outliers(df, column, threshold,id_key="id"):
    # Identify outliers
    is_outlier = (df[column] - df[column].mean()).abs() > threshold * df[column].std()
    
    # Turn outliers to NaN
    df_cleaned = df.copy()
    df_cleaned.loc[is_outlier, column] = np.nan
    df_cleaned.loc[df_cleaned[column]<0,column]=np.nan
    # Interpolate missing values
    df_cleaned[column] = df_cleaned.groupby(id_key)[column].transform(lambda x: x.interpolate())
    #ffill and bfill
    df_cleaned[column] = df_cleaned[column].bfill().ffill()

    

    return df_cleaned



def cleaning(gps:pd.DataFrame,rpe=None,th_long=300,th_outliers=3,id_key="id"):

    # TODO : 
    # - remove the with constant distance
    # - be more specific on the missing data to be better in interpolation 
    # ie removing in the session the part that are null for a long time
    print("shape before cleaning : ",gps.shape)
    print('nb of sessions : ',gps["id"].nunique())
    #remove the long sequences of nan values in watts and hr
    gps = remove_long_nan_sequences(gps, 'stream_watts', th_long,id_key=id_key)
    gps = remove_long_nan_sequences(gps, 'stream_heartrate', th_long,id_key=id_key)
    print("shape after removing the long sequences of nan values : ",gps.shape)
    print('nb of sessions : ',gps["id"].nunique())
    # isnull = gps[["stream_heartrate","stream_watts","stream_distance"]].isnull().groupby(gps["id"]).mean().reset_index()
    # cleaned = gps[gps["id"].isin(isnull[isnull["stream_watts"] < 0.75]["id"])]


    # same for hr
    # cleaned = cleaned[cleaned["id"].isin(isnull[isnull["stream_heartrate"] < 0.75]["id"])]
    # print("shape after removing the sessions with more than 50% of missing values : ",cleaned.shape)
    # fill the missing values with an interpolation within the session

    cleaned = gps.copy()
    if cleaned.index.duplicated().any():
        raise ValueError("Duplicate indices found in the DataFrame")

    # number of missing values group by id
    isnull = cleaned.isnull().groupby("id").sum()
    print("number of missing values group by id : ",isnull.sum())

    cleaned = interpolate_missing_values(cleaned, 'stream_heartrate',id_key=id_key)
    cleaned = interpolate_missing_values(cleaned, 'stream_watts',id_key=id_key)
    # cleaned = interpolate_missing_values(cleaned, 'stream_distance',id_key=id_key)

    print("shape after interpolation : ",cleaned.shape)
    print('nb of sessions : ',cleaned["id"].nunique())


    #fill the remaining missing values with the next value for the 3 columns
    isnull = cleaned.isnull().groupby("id").sum()
    print("number of missing values group by id after interpolation : ",isnull.sum())
    print('nb of sessions : ',cleaned["id"].nunique())


    # remove the outliers
    cleaned = filter_outliers(cleaned, 'stream_watts', th_outliers)
    cleaned = filter_outliers(cleaned, 'stream_heartrate', th_outliers)
    print("shape after removing the outliers : ",cleaned.shape)
    print('nb of sessions : ',cleaned["id"].nunique())

    # mean_watts = cleaned[['id','stream_watts']].groupby("id").agg({"stream_watts":np.nanmean})
    # cleaned = cleaned[cleaned["id"].isin(mean_watts[mean_watts>20].index)]
    # print("shape after removing the id with less than 20 watts : ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())
    # print("number of id with less than 20 watts : ",mean_watts[mean_watts<20].shape[0])
    
    # # same for the hr < 50
    # mean_hr = cleaned.groupby("id").mean()["stream_heartrate"]
    # cleaned = cleaned[cleaned["id"].isin(mean_hr[mean_hr>50].index)]
    # print("shape after removing the id with less than 50 hr : ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())
    
    # cleaned = cleaned.merge(rpe, how="inner", left_on="id", right_on="id_session"

    # remove the frames with 0 watts
    # cleaned = cleaned[cleaned["stream_watts"]!=0]
    # print("shape after removing the 0 watts frames: ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())
    
    #remove the sessions with less than 600 frames
    # count = cleaned.groupby("id").count()["tps"]
    # cleaned = cleaned[cleaned["id"].isin(count[count>300].index)]
    # print("shape after removing the sessions with less than 300 frames: ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())
    #remove sessions with negative watts
    # neg_id = cleaned[cleaned["stream_watts"]<0]["id"].unique()
    # cleaned = cleaned[~cleaned["id"].isin(neg_id)]
    # print("shape after removing the sessions with negative watts: ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())

    # remove sessions with watts values > 1500
    # high_id = cleaned[cleaned["stream_watts"]>1500]["id"].unique()
    # cleaned = cleaned[~cleaned["id"].isin(high_id)]
    # print("shape after removing the sessions with high watts: ",cleaned.shape)
    # print('nb of sessions : ',cleaned["id"].nunique())

    dates = rpe[["id_session","dt_session"]].drop_duplicates()
    cleaned = cleaned.merge(dates, how="inner", left_on="id", right_on="id_session")
    cleaned.sort_values(by=['dt_session',"tps"],inplace=True)

    return cleaned


def cleaning_all_ath(gps_all:pd.DataFrame,meta_data:pd.DataFrame):
    cleaned_all = pd.DataFrame()
    for ath in tqdm(gps_all["ath_id"].unique()):
        gps = gps_all[gps_all["ath_id"]==ath]
        meta_selected = meta_data[meta_data["ath_id"]==ath]
        # print(gps.head())
        cleaned = new_cleaning(gps,meta_selected,th_long=120,th_outliers=3,small_session = 600,id_key='id')
        cleaned_all = pd.concat([cleaned_all,cleaned])
    return cleaned_all

def normalize_all_ath(cleaned:pd.DataFrame,meta_data:pd.DataFrame,fields=["stream_watts"],period=60,rolling=False,max_ppr=1000,method_hr="mean",id_key="id"):
    ath_ids = cleaned["ath_id"].unique()
    normalized_all = pd.DataFrame()
    norm_data_res = pd.DataFrame()
    for ath in tqdm(ath_ids):
        meta_ath = meta_data[meta_data["ath_id"]==ath].reset_index()
        gps = cleaned[cleaned["ath_id"]==ath].reset_index()
        # gps=gps.merge(meta_ath[id_key,"date"],how="inner",left_on=id_key,right_on=id_key)
        # gps.sort_values(by=['date',"tps"],inplace=True)
        # gps.drop(columns=["date"],inplace=True)
        #check no duplicates for id_key
        normalized,norm_data = new_normalize(gps,meta_ath,fields=["stream_watts","stream_heartrate"],period=period,rolling=rolling,max_ppr=max_ppr,method_hr=method_hr,id_key=id_key)
        print(normalized.shape)
        normalized['ath_id'] = ath
        norm_data['ath_id'] = ath
        normalized_all = pd.concat([normalized_all,normalized])
        norm_data_res = pd.concat([norm_data_res,norm_data])
    return normalized_all[[id_key,'ath_id',"tps","stream_watts","stream_heartrate"]],norm_data_res
