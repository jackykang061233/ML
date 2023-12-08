import typing as t

import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import ROOT, config, TRAINED_MODEL_DIR, PACKAGE_ROOT
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs


def load_save_file(pipeline_file_name):
    """
    Load a saved machine learning pipeline from a file.

    Parameters:
    - pipeline_file_name (str): The name of the file containing the saved pipeline.

    Returns:
    Pipeline: A trained pipeline
    """
    _pipe = load_pipeline(file_name=pipeline_file_name)
    return _pipe

def make_prediction(*, pipeline_file_name: str, predict_data: t.Union[pd.DataFrame, dict]) -> dict:
    """
    Make predictions using a pipeline.

    Parameters:
    - pipeline_file_name (str): The name of the file containing the saved pipeline.
    - predict_data (pd.DataFrame or dict): The data for making predictions. Must have the features required by the pipeline.

    Returns:
    np.ndarray: predictions (0 or 1) based on the specified precision-recall threshold.

    """
    
    to_drop = config.log_config.to_drop

    predict_data = predict_data.drop(to_drop, axis=1)
    
    _pipe = load_save_file(pipeline_file_name)

    predictions = _pipe.predict_proba(predict_data)
    predictions = np.where(predictions[:, 1]>=config.log_config.precision_recall_threshold, 1, 0)
    
    return predictions

def prediction(pipeline_file_name=None):
    """
    Make predictions on a dataset and save the results to a CSV file. If the format of input data is not correct, errors will be printed out

    Parameters:
    - pipeline_file_name (str): The name of the file containing the saved pipeline.

    Examples:
    prediction(pipeline_file_name='xgboost_output_v0.0.1.pkl')

    """
    if not pipeline_file_name:
        newest_model = PACKAGE_ROOT / 'newest_model.txt'
        with open(newest_model, 'r') as f:
            pipeline_file_name = f.read()
    predict_data = pd.read_csv(str(ROOT)+ config.app_config.predict_path)
    predict_data, errors = validate_inputs(input_data=predict_data)

    if errors:
        print(errors)
        print('The data has an incorrect format.')
        return 
    results = make_prediction(pipeline_file_name=pipeline_file_name, predict_data=predict_data)
    results = pd.DataFrame({'txkey': predict_data['txkey'].values, 'pred': results})

    results.to_csv(f'{str(ROOT)}/submissions/{config.app_config.pipeline_save_file}{_version}.csv', index=False)
