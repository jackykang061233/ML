# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model.config.core import config, ROOT
from model.pipeline import pipeline
from model.evaluate import grid_search_cv
from model.processing.data_manager import save_pipeline

# scikit-learn
from sklearn.model_selection import train_test_split

# mlflow
from mlflow.models.signature import infer_signature
import mlflow

from utils import accuracy, precision, recall, f1, auc

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
    
def train():
    X_train, X_test, y_train, y_test = data_prep()

    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')

    experiment_name = config.mlflow_config.experiment_name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, tags=config.mlflow_config.experiment_tags)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name='train') as run:
        models = {'logistic_regression': dict(config.log_config.logistic),
              'random_forest': dict(config.log_config.random_forest),
              'xgboost': dict(config.log_config.xgb)}

        pipe = pipeline(X_train.columns)
        model = pipe.fit(X_train, y_train)

        # get selected features' names
        select_k_best = pipe.named_steps['feature_selection'].named_transformers_['selected_columns']
        cols_idxs = select_k_best.get_support(indices=True).to_list()
        selected_columns = [col for index, col in enumerate(config.log_config.categorical_features) if index in cols_indexs]
        mlflow.log_params({'selected_features': selected_columns})

        predictions = model.predict(X_test)

        accuracy_score = accuracy(y_test, predictions)
        precision_score = precision(y_test, predictions)
        recall_score = recall(y_test, predictions)
        f1_score = f1(y_test, predictions)
        auc_score = auc(y_test, predictions)

        mlflow.log_metrics({
            'accuracy': accuracy_score, 
            'precision': precision_score, 
            'recall': recall_score, 
            'f1': f1_score, 
            'auc': auc_score
        })

        if config.log_config.use_sampling:
            mlflow.log_params({'smote': dict(config.log_config.smote)})
        mlflow.log_params({'config.log_config.used_model': models[config.log_config.used_model]})
        

        signature = infer_signature(X_train, predictions )
        mlflow.sklearn.log_model(model, 'model', signature=signature)
                           
        
    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)



def train_grid_search():
    X_train, X_test, y_train, y_test = data_prep()
    
    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')
    
    experiment_name = config.mlflow_config.experiment_name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, tags=config.mlflow_config.experiment_tags)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name='train') as run:
        model, params = grid_search_cv(X_train, y_train)

        pipe = pipeline(X_train.columns)
        model = pipe.fit(X_train, y_train)

        # get selected features' names
        select_k_best = pipe.named_steps['feature_selection'].named_transformers_['selected_columns']
        cols_idxs = select_k_best.get_support(indices=True).to_list()
        selected_columns = [col for index, col in enumerate(config.log_config.categorical_features) if index in cols_indexs]
        mlflow.log_params({'selected_features': selected_columns})

        predictions = model.predict(X_test)

        accuracy_score = accuracy(y_test, predictions)
        precision_score = precision(y_test, predictions)
        recall_score = recall(y_test, predictions)
        f1_score = f1(y_test, predictions)
        auc_score = auc(y_test, predictions)

        mlflow.log_metrics({
            'accuracy': accuracy_score, 
            'precision': precision_score, 
            'recall': recall_score, 
            'f1': f1_score, 
            'auc': auc_score
        })

        if config.log_config.use_sampling:
            mlflow.log_params({'smote': dict(config.log_config.smote)})
        mlflow.log_params({'config.log_config.used_model': params})

        signature = infer_signature(X_train, predictions )
        mlflow.sklearn.log_model(model, 'model', signature=signature)
                           
        
    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)



if __name__ == '__main__':
    train()

