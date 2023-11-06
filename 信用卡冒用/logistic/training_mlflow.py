`# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from logistic.config.core import config, ROOT
from logistic.pipeline import pipe
# from logistic.predict import evaluation
from logistic.processing.data_manager import save_pipeline

# scikit-learn
from sklearn.model_selection import train_test_split

# mlflow
from mlflow.models.signature import infer_signature

from utils import accuracy, precision, recall, f1, auc


def train():
    df = pd.read_csv(str(ROOT)+config.app_config.training_data)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    # take only partial data to train
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=config.log_config.samples_to_train_ratio, stratify=y_train, random_state=config.log_config.random_state)

    
    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')

    experiment_name = config.mlflow_config.experiment_name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, tags=config.mlflow_config.experiment_tags)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start(run_name='train') as run:
        model = pipe.fit(X_train, y_train)

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

        mlflow.log_params(config.log_config.smote.model_dump())
        mlflow.log_params(config.log_config.logistic.model_dump())

        signature = infer_signature(model_input=X_train, model_output=predictions)
        mlflow.sklearn.log_model(model, 'model', signature)
                           

        
    print('--------END TRAINING--------')
    save_pipeline(pipeline_to_save=model)

    # evaluation(X_test, y_test)

if __name__ == '__main__':
    train()

