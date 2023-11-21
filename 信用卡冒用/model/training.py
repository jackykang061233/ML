# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model.config.core import config, ROOT, TRAINED_MODEL_DIR
from model.pipeline import pipeline
from model.evaluate import evaluation, cross_validation, grid_search_cv
from model.processing.data_manager import save_pipeline
from model import __version__ as _version

# scikit-learn
from sklearn.model_selection import train_test_split

def data_prep():
    df = pd.read_csv(str(ROOT)+config.app_config.training_data)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    # take only partial data to train
    if config.log_config.samples_to_train_ratio==1:
        pass
    else:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-config.log_config.samples_to_train_ratio, stratify=y_train, random_state=config.log_config.random_state)
        
    return X_train, X_test, y_train, y_test

def custom_val_set():
    df = pd.read_csv(str(ROOT)+config.app_config.training_data)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X_train = df.drop(to_drop+[target], axis=1)
    y_train = df[target]

    val = pd.read_csv(str(ROOT)+config.app_config.val_data)
    X_test = val.drop(to_drop+[target], axis=1)
    y_test = val[target]

    return X_train, X_test, y_train, y_test
    
def train():
    X_train, X_test, y_train, y_test = custom_val_set()
    # X_train, X_test, y_train, y_test = data_prep()

    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')
    pipe = pipeline(X_train.columns)
    model = pipe.fit(X_train, y_train)

    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)

    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

def train_cross_validation():
    X_train, X_test, y_train, y_test = data_prep()
    
    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')
    model = cross_validation(X_train, y_train)
    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)

    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

def train_grid_search():
    X_train, X_test, y_train, y_test = data_prep()
    
    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')
    model, params, pipe = grid_search_cv(X_train, y_train)
    model.fit(X_train, y_train)
    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)

    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

    grid_search_path = str(TRAINED_MODEL_DIR) + '/grid_search/' + f"{config.app_config.pipeline_save_file}.txt"
    with open(grid_search_path, 'w') as f:
        for param_name, param in params.items():
            f.write(f'{param_name}: {param}'+'\n')


if __name__ == '__main__':
    train()

