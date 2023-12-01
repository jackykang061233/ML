import pytest
import pandas as pd

@pytest.fixture()
def sample_input_data():
    return pd.DataFrame({'not_important': [1, 2, 3, 4, 5, 6], 'loctm': [0, 235959, 120000, 180000, 205159, 71232]})

@pytest.fixture()
def sample_train_data():
    pass