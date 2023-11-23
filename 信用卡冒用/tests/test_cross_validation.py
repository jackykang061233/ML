import pytest

from model.training import train_cross_val

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def test_train_cross_val():
    train_cross_val()
