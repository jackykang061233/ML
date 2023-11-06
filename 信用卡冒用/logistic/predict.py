import typing as t

import numpy as np
import pandas as pd

from logistic import __version__ as _version
from logistic.config.core import config, TRAINED_MODEL_DIR
from logistic.processing.data_manager import load_pipeline
from utils import accuracy, precision, recall, f1, auc


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)

def prediction(*, test_data: t.Union[pd.DataFrame, dict]) -> dict:
    data = pd.DataFrame(test_data)
    results = {"predictions": None, "version": _version}

    predictions = _pipe.predict(test_data)
    results = {"predictions": predictions, "version": _version}

    return results


def evaluation(*, test_data: t.Union[pd.DataFrame, dict], y_test: t.Union[pd.DataFrame, dict]):
    data = pd.DataFrame(test_data)
    results = {"predictions": None, "version": _version}

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
    