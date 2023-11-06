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

def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_save, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
    