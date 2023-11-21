import pytest
from model.evaluate import evaluation
import pandas as pd
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, ROOT

def test_evaluation():
    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    test_data = pd.read_csv(str(ROOT)+config.app_config.training_data)
    prediction(pipeline_file_name)

