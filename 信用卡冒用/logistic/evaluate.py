import typing as t

import numpy as np
import pandas as pd

from logistic import __version__ as _version
from logistic.config.core import config, TRAINED_MODEL_DIR
from logistic.processing.data_manager import load_pipeline
from utils import accuracy, precision, recall, f1, auc


def load_save_file(pipeline_file_name):
    _pipe = load_pipeline(file_name=pipeline_file_name)
    return _pipe
    
def evaluation(*, pipeline_file_name: str, test_data: t.Union[pd.DataFrame, dict], y_test: t.Union[pd.DataFrame, dict]):
    _pipe = load_save_file(pipeline_file_name)
    data = pd.DataFrame(test_data)

    predictions = _pipe.predict(data)
    
    to_write = [f'accuracy {accuracy(y_test, predictions)}', 
                f'precision {precision(y_test, predictions)}', 
                f'recall {recall(y_test, predictions)}', 
                f'f1 {f1(y_test, predictions)}', 
                f'auc {auc(y_test, predictions)}']
    
    to_write_path = str(TRAINED_MODEL_DIR) + '/metric/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    with open(to_write_path, 'w') as f:
        for w in to_write:
            f.write(w+'\n')