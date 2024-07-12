import pytest
import pandas as pd
import numpy as np
from utils.transformation import get_norm_data, new_normalize  # replace 'your_module' with the actual module name

@pytest.fixture
def setup_data():
    # Create sample data for meta_data
    data_meta = {
        'id_session': np.arange(1, 21),
        'date': pd.date_range(start='2022-03-01', periods=20, freq='D').strftime('%Y-%m-%d %H:%M:%S'),
        'rpe': np.random.randint(1, 10, size=20),
        'sport': ['Vélo - Route' if i % 2 == 0 else 'Vélo - Piste' for i in range(20)],
        'ath_id': [15600] * 20
    }
    meta_data = pd.DataFrame(data_meta)

    # Create sample data for cleaned
    sessions = np.repeat(np.arange(1, 21), 10)
    tps = np.tile(np.arange(10), 20)
    stream_watts = np.random.randint(50, 150, size=200)
    stream_heartrate = np.random.randint(100, 180, size=200)
    dt_session = np.repeat(pd.date_range(start='2019-01-01', periods=20, freq='D').strftime('%Y-%m-%d %H:%M:%S'), 10)
    
    data_cleaned = {
        'id_session': sessions,
        'tps': tps,
        'stream_watts': stream_watts,
        'stream_heartrate': stream_heartrate,
        'date': dt_session
    }
    cleaned = pd.DataFrame(data_cleaned)

    # Create sample data for norm_data
    data_norm = {
        'id_session': np.arange(1, 21),
        'ppr': np.random.uniform(300, 350, size=20),
        'ma_hr': np.random.uniform(150, 160, size=20),
        'roll_std_hr': np.random.uniform(1, 20, size=20),
        'roll_max_hr': np.random.uniform(150, 170, size=20),
        'rpe': np.random.randint(1, 10, size=20),
        'date': pd.date_range(start='2022-03-01', periods=20, freq='D').strftime('%Y-%m-%d'),
        'sport': ['Vélo - Route' if i % 2 == 0 else 'Vélo - Piste' for i in range(20)]
    }
    norm_data = pd.DataFrame(data_norm)

    return meta_data, cleaned, norm_data

def test_get_norm_data(setup_data):
    meta_data, cleaned, _ = setup_data
    id_key = 'id_session'
    result = get_norm_data(cleaned, meta_data, id_key=id_key)
    # Add assertions to check if the result is as expected
    assert 'ma_hr' in result.columns
    assert 'roll_std_hr' in result.columns
    assert 'roll_max_hr' in result.columns
    assert 'ppr' in result.columns
    # check no nan values
    assert result.isnull().sum().sum() == 0

def test_new_normalize(setup_data):
    meta_data, cleaned, _ = setup_data
    fields = ["stream_watts", "stream_heartrate"]
    keep_cols = []#["ppr", "ma_hr", "roll_std_hr", "roll_max_hr", "rpe", "date", "sport"]
    id_key = 'id_session'
    normalized_df, norm_data = new_normalize(cleaned, meta_data, fields=fields, keep_cols=keep_cols, id_key=id_key)
    
    # Add assertions to check if the normalized data is as expected
    assert 'stream_watts' in normalized_df.columns
    assert 'stream_heartrate' in normalized_df.columns
    assert id_key in normalized_df.columns
    assert 'tps' in normalized_df.columns

    # norm data is okay 
    assert 'ma_hr' in norm_data.columns
    assert 'roll_std_hr' in norm_data.columns
    assert 'roll_max_hr' in norm_data.columns
    assert 'ppr' in norm_data.columns
    assert 'rpe' in norm_data.columns
    assert 'date' in norm_data.columns
    assert 'sport' in norm_data.columns

    #shape is okay
    assert normalized_df.shape[0] == cleaned.shape[0]
    assert normalized_df.shape[1] == 2+ len(fields) + len(keep_cols)
    
    # check no nan values
    assert normalized_df.isnull().sum().sum() == 0
