from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load
import pydantic


PACKAGE_ROOT = Path(__file__).parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
DATASET_DIR = ROOT / 'data'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'train_models'

class SmoteConfig(BaseModel):
    sampling_strategy: float
    k_neighbors: int

class LogisticRegressionConfig(BaseModel):
    max_iter: int
    solver: str
    n_jobs: int

class RandomForestConfig(BaseModel):
    bootstrap: bool
    random_state: int
    class_weight: Dict[int, float]
    n_jobs: int

class AppConfig(BaseModel):
    """
    Application-level config
    """
    package_name: str
    training_data: str
    pipeline_save_file: str
    predict_path: str

class LogConfig(BaseModel):
    target: str
    used_model: str
    samples_to_train_ratio: float
    to_drop: List[str]
    random_state: int
    test_size: float
    vars_with_na: List[str]
    time_transform: str
    features: List[str]
    smote: SmoteConfig
    logistic: LogisticRegressionConfig
    random_forest: RandomForestConfig 

class MLflowConfig(BaseModel):
    experiment_name: str
    experiment_tags: Dict[str, str]

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    log_config: LogConfig
    mlflow_config: MLflowConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    
    _config = Config(
        app_config=AppConfig(**parsed_config['app_config'].data),
        log_config=LogConfig(**parsed_config['log_config'].data),
        mlflow_config=MLflowConfig(**parsed_config['mlflow_config'].data)
    )

    return _config

config = create_and_validate_config()