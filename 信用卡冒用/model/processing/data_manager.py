import typing as t
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline


from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, ML_ROOT, ROOT

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

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    test = pd.read_csv(str(ROOT)+config.app_config.val_data)
    X_test = test.drop(to_drop+[target], axis=1)
    y_test = test[target]

    return X_train, X_val, X_test, y_train, y_val, y_test

def public_as_val():
    df = pd.read_csv(str(ROOT)+config.app_config.training_data)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]


    test = pd.read_csv(str(ROOT)+config.app_config.val_data)
    X_test = test.drop(to_drop+[target], axis=1)
    y_test = test[target]

    return X, X_test, y, y_test

def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_save, save_path)

def load_pipeline(*, file_name: str, mlflow: bool = False) -> Pipeline:
    if mlflow:
        file_path = ML_ROOT / file_name
    else:
        file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
    