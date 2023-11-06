# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from logistic.config.core import config, ROOT
from logistic.pipeline import pipe
from logistic.predict import evaluation
from logistic.processing.data_manager import save_pipeline

# scikit-learn
from sklearn.model_selection import train_test_split


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
    model = pipe.fit(X_train, y_train)
    print('--------END TRAINING--------')
    # save_pipeline(pipeline_to_save=model)

    evaluation(test_data=X_test, y_test=y_test)

if __name__ == '__main__':
    train()

