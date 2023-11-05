# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config.core import config, ROOT
from pipeline import pipe

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score,roc_auc_score, accuracy_score


def train(train_ratio=config.log_config.samples_to_train_ratio):
    df = pd.read_csv(str(ROOT)+config.app_config.training_data)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    # take only partial data to train
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=train_ratio, stratify=y_train, random_state=config.log_config.random_state)

    
    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')
    model = pipe.fit(X_train, y_train)
    print('--------END TRAINING--------')

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)

    print(f'accuracy {accuracy}')
    print(f'precision {precision}')
    print(f'recall {recall}')
    print(f'f1 {f1}')
    print(f'auc {auc}')

if __name__ == '__main__':
    train()

