import pytest
from model.training import train
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def test_train():
    train()

    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    assert os.path.isfile(save_path)