import pytest
from model.predict import prediction
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def test_prediction():
    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    prediction(pipeline_file_name)
