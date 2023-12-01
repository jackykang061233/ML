import pytest
from model.training_mlflow import mlflow_train, InvalidTrainTypeException
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def test_train_mlflow():
    mlflow_train(train_type='train')

    # save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    # save_path = TRAINED_MODEL_DIR / save_file_name
    # assert os.path.isfile(save_path)

def test_gridsearch_mlflow():
    mlflow_train(train_type='grid_search')
    
    # save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    # save_path = TRAINED_MODEL_DIR / save_file_name
    # assert os.path.isfile(save_path)

def test_cv_mlflow():
    mlflow_train(train_type='cross_validation')


def test_invalid_train_type():
    with pytest.raises(InvalidTrainTypeException):
        mlflow_train(train_type='grarch')