import pytest
import pandas as pd
import numpy as np
from utils.transformation import remove_long_nan_sequences, interpolate_missing_values, filter_outliers, new_cleaning, remove_long_nan_sequences_v2,remove_long_nan_sequences_v3

@pytest.fixture
def setup_df():
    df = pd.DataFrame({
        'id': np.repeat(np.arange(6), 10),
        'tps': np.tile(np.arange(10), 6),
        'column': np.array([np.nan, np.nan, np.nan, 3, np.nan, 5, np.nan, np.nan, 8, 9] * 6)
    })
    return df

def test_remove_long_nan_sequences(setup_df):
    df_cleaned = remove_long_nan_sequences(setup_df, 'column', 1)
    assert df_cleaned['column'].isna().sum() == 6
    assert df_cleaned.shape == (30, 3)

def test_remove_long_nan_sequences_v2(setup_df):
    df_cleaned = remove_long_nan_sequences_v2(setup_df, 'column', 1)
    assert df_cleaned['column'].isna().sum() == 6
    assert df_cleaned.shape == (30, 3)

def test_remove_long_nan_sequences_v3(setup_df):
    df_cleaned = remove_long_nan_sequences_v3(setup_df, 'column', 1)
    assert df_cleaned['column'].isna().sum() == 6
    assert df_cleaned.shape == (30, 3)

def test_interpolate_missing_values(setup_df):
    df_interpolated = interpolate_missing_values(setup_df, 'column')
    assert df_interpolated['column'].isna().sum() == 0

def test_filter_outliers(setup_df):
    setup_df['column'] = setup_df['column'].replace(1, 100)
    df_cleaned = filter_outliers(setup_df, 'column', 2)
    assert 100 not in df_cleaned['column']

@pytest.fixture
def setup_gps_rpe():
    gps = pd.DataFrame({
        'id': np.repeat(np.arange(6), 11),
        'tps': np.tile(np.arange(11), 6),
        'stream_watts': np.array([np.nan, np.nan, np.nan, -100, 95, 100, np.nan, np.nan, 100, np.nan, 95] * 6),
        'stream_heartrate': np.array([np.nan, np.nan, np.nan, 120, -10, 131, np.nan, np.nan, 127, np.nan, 128] * 6),
    })
    gps['stream_watts'] = gps['stream_watts'] + gps['id'] * 10
    gps['stream_heartrate'] = gps['stream_heartrate'] + gps['id'] * 10
    rpe = pd.DataFrame({
        'id_session': np.arange(6),
        'dt_session': ['2021-01-05', '2021-01-03', '2021-01-01', '2021-01-06', '2021-01-02', '2021-01-04'],
    })
    return gps, rpe

def test_new_cleaning(setup_gps_rpe):
    gps, rpe = setup_gps_rpe
    cleaned = new_cleaning(gps, rpe, th_long=1, th_outliers=1, id_key='id')
    assert 'date' in cleaned.columns
    assert 'id_session' in cleaned.columns
    assert 'tps' in cleaned.columns
    assert 'stream_watts' in cleaned.columns
    assert 'stream_heartrate' in cleaned.columns
    assert cleaned.shape == (36, 5)
    assert cleaned['id_session'].nunique() == 6
    assert cleaned.isnull().sum().sum() == 0
    assert -100 not in cleaned['stream_watts']
    assert -10 not in cleaned['stream_heartrate']
    assert cleaned['date'].iloc[0] == '2021-01-01'
