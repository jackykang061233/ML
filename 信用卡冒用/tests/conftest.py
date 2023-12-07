import pytest
import pandas as pd

@pytest.fixture()
def sample_input_data():
    return pd.DataFrame({'not_important': [1, 2, 3, 4, 5, 6], 'loctm': [0, 235959, 120000, 180000, 205159, 71232]})

@pytest.fixture()
def sample_train_data():
    test = pd.read_csv('data/test.csv')
    return test

@pytest.fixture()
def sample_predict_data():
    input_data = pd.DataFrame({
            'txkey': ['1', '2', '3'],
            'locdt': [1, 2, 3],
            'loctm': [100000, 200000, 300000],
            'chid': ['A', 'B', 'C'],
            'cano': ['111', '222', '333'],
            'contp': [1, 2, 3],
            'etymd': [1.0, 2.0, 3.0],
            'mchno': ['M1', 'M2', 'M3'],
            'acqic': ['A1', 'A2', 'A3'],
            'mcc': [100.0, 200.0, 300.0],
            'conam': [1000.0, 2000.0, 3000.0],
            'ecfg': [1, 0, 1],
            'insfg': [1, 0, 1],
            'iterm': [1.0, 2.0, 3.0],
            'bnsfg': [1, 0, 1],
            'flam1': [1, 0, 1],
            'stocn': [100.0, 200.0, 300.0],
            'scity': [100.0, 200.0, 300.0],
            'stscd': [1.0, 2.0, 3.0],
            'ovrlt': [1, 0, 1],
            'flbmk': [1, 0, 1],
            'hcefg': [1.0, 2.0, 3.0],
            'csmcu': [100.0, 200.0, 300.0],
            'csmam': [1000, 2000, 3000],
            'flg_3dsmk': [1, 0, 1],
            'label': [0, 1, 0]
        })
    return input_data