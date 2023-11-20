import typing as t

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from model import __version__ as _version
from model.pipeline import pipeline
from model.config.core import config, TRAINED_MODEL_DIR
from model.processing.data_manager import load_pipeline
from utils import accuracy, precision, recall, f1, auc


def load_save_file(pipeline_file_name):
    _pipe = load_pipeline(file_name=pipeline_file_name)
    return _pipe
    
def evaluation(*, pipeline_file_name: str, test_data: t.Union[pd.DataFrame, dict], y_test: t.Union[pd.DataFrame, dict]):
    _pipe = load_save_file(pipeline_file_name)
    data = pd.DataFrame(test_data)

    predictions = _pipe.predict_proba(data)
    predictions = np.where(predictions[:, 1]>=config.log_config.precision_recall_threshold, 1, 0)
    
    to_write = [f'accuracy {accuracy(y_test, predictions)}', 
                f'precision {precision(y_test, predictions)}', 
                f'recall {recall(y_test, predictions)}', 
                f'f1 {f1(y_test, predictions)}', 
                f'auc {auc(y_test, predictions)}']
    
    to_write_path = str(TRAINED_MODEL_DIR) + '/metric/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    with open(to_write_path, 'w') as f:
        for w in to_write:
            f.write(w+'\n')

def cross_validation(X_train, y_train):
    skf = StratifiedKFold(**dict(config.cv_config.stratifiedkfold))
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    auc_lst = []

    for train, test in skf.split(X_train, y_train):
        pipe = pipeline(X_train.columns)
        model = pipe.fit(X_train.iloc[train], y_train.iloc[train])
        predictions = model.predict_proba(X_train.iloc[test])
        predictions = np.where(predictions[:, 1]>=config.log_config.precision_recall_threshold, 1, 0)
        
        accuracy_lst.append(accuracy(y_train.iloc[test], predictions))
        precision_lst.append(precision(y_train.iloc[test], predictions))
        recall_lst.append(recall(y_train.iloc[test], predictions))
        f1_lst.append(f1(y_train.iloc[test], predictions))
        auc_lst.append(auc(y_train.iloc[test], predictions))

    to_write = [f'accuracy {np.mean(accuracy_lst)}', 
              f'precision {np.mean(precision_lst)}', 
              f'recall {np.mean(recall_lst)}', 
              f'f1 {np.mean(f1_lst)}', 
              f'auc {np.mean(auc_lst)}']
    to_write_path = str(TRAINED_MODEL_DIR) + '/metric/cv/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    with open(to_write_path, 'w') as f:
        for w in to_write:
            f.write(w+'\n')
    return model

def grid_search_cv(X_train, y_train):
    # models = {'Logistic Regression': dict(config.cv_config.logistic),
    #           'Random Forest': dict(config.cv_config.random_forest),
    #           'Xgboost':dict(config.cv_config.xgb)}
    models = {'random_forest': dict(config.cv_config.random_forest),}

    skf = StratifiedKFold(**dict(config.cv_config.stratifiedkfold))

    pipe = pipeline(X_train.columns)
    grid_search = GridSearchCV(pipe, param_grid=models[config.log_config.used_model], scoring='f1', cv=skf)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    return best_model, best_parameters, pipe

def find_best_threshold(*, pipeline_file_name: str, test_data: t.Union[pd.DataFrame, dict], y_test: t.Union[pd.DataFrame, dict]):
    _pipe = load_save_file(pipeline_file_name)
    data = pd.DataFrame(test_data)

    f1_score = []
    prediction = _pipe.predict_proba(data)
    thresholds = [round(x, 2) for x in np.arange(0.95, -0.05, -0.05)]

    for t in thresholds:
        predictions = np.where(prediction[:, 1]>=t, 1, 0)
        score = f1(y_test, predictions)
        f1_score.append(score)
    best_score_index = np.array(f1_score).argmax()
    best_threshold = thresholds[best_score_index]
    return best_threshold, f1_score[best_score_index]
        
    
    
    
    
    


    
    
    