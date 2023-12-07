import pytest
from model.training import train
import os

from model import __version__ as _version
from model.config.core import TRAINED_MODEL_DIR, config
from model.exception import InvalidTrainTypeException

def test_train():
    train(train_type='train')

    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    assert os.path.isfile(save_path)

def test_train_cross_val():
    train(train_type='cross_validation')
    
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    assert os.path.isfile(save_path)
    
    cv_metrics_path = str(TRAINED_MODEL_DIR) + '/metric/cv/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    assert os.path.isfile(cv_metrics_path)
    
def test_gridsearch():
    train(train_type='grid_search')
    
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    assert os.path.isfile(save_path)
    
    grid_search_path = str(TRAINED_MODEL_DIR) + '/grid_search/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    assert os.path.isfile(grid_search_path)

def test_train_invalid_type():
    with pytest.raises(InvalidTrainTypeException):
        train(train_type='invalid_type')