import pytest
from model.evaluate import evaluation
import pandas as pd
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, ROOT

def test_evaluation(sample_train_data):
    pipeline_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    test_data = sample_train_data
    to_drop = config.log_config.to_drop
    target = config.log_config.target
    X_test = test_data.drop(to_drop+[target], axis=1)
    y_test = test_data[target]

    evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

    save_file_path = str(TRAINED_MODEL_DIR) + '/metric/' + f"{config.app_config.pipeline_save_file}{_version}.txt"
    assert os.path.isfile(save_file_path)

    with open(save_file_path, 'r') as f:
        content = f.read()
        assert "accuracy" in content
        assert "precision" in content
        assert "recall" in content
        assert "f1" in content
        assert "auc" in content

