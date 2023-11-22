import pytest
from model.evaluate import evaluation
import pandas as pd
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, ROOT

def test_evaluation():
    pipeline_file_name = "random_forest_output_v0.0.1.pkl"
    test_data = pd.read_csv(str(ROOT)+config.app_config.val_data)
    to_drop = config.log_config.to_drop
    target = config.log_config.target
    X_test = test_data.drop(to_drop+[target], axis=1)
    y_test = test_data[target]

    evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

