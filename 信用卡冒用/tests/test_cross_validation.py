import pytest
from model.evaluate import cross_validation
from model.training import data_prep, custom_val_set
import os

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def test_train_cross_val():
    X_train, _, y_train, _ = custom_val_set()
    cross_validation(X_train, y_train)
