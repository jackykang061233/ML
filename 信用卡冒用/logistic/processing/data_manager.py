import typing as t
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline


from logistic import __version__ as _version
from logistic.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f'{DATASET_DIR}/{file_name}'))
    return df