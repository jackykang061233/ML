import typing as t

import numpy as np
import pandas as pd

from logistic import __version__ as _version
from logistic.config.core import ROOT, config, TRAINED_MODEL_DIR
from logistic.processing.data_manager import load_pipeline


def load_save_file(pipeline_file_name):
    _pipe = load_pipeline(file_name=pipeline_file_name)
    return _pipe

def make_prediction(*, pipeline_file_name: str, predict_data: t.Union[pd.DataFrame, dict]) -> dict:
    to_drop = config.log_config.to_drop

    predict_data = predict_data.drop(to_drop, axis=1)
    
    _pipe = load_save_file(pipeline_file_name)
    data = pd.DataFrame(predict_data)

    predictions = _pipe.predict(predict_data)

    return predictions

def prediction(pipeline_file_name):
    predict_data = pd.read_csv(str(ROOT)+ config.app_config.predict_path)

    results = make_prediction(pipeline_file_name=pipeline_file_name, predict_data=predict_data)
    results = pd.DataFrame({'txkey': predict_data['txkey'].values, 'pred': results})

    results.to_csv(f'{str(ROOT)}/submissions/{config.app_config.pipeline_save_file}{_version}.csv', index=False)


if __name__ == '__main__':
    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    prediction(pipeline_file_name)
    