from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, validator
from typing_extensions import Annotated
from strictyaml import YAML, load
import pydantic


PACKAGE_ROOT = Path(__file__).parent.parent
ROOT = PACKAGE_ROOT.parent
ML_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
DATASET_DIR = ROOT / 'data'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'train_models'

# App
class AppConfig(BaseModel):
    """
    Application-level config
    """
    package_name: str
    training_data: str
    val_data: str
    test_data: str
    pipeline_save_file: str
    predict_path: str


# Model config
class SmoteConfig(BaseModel):
    sampling_strategy: float
    k_neighbors: int

class LogisticRegressionConfig(BaseModel):
    max_iter: int
    solver: str
    n_jobs: int

class RandomForestConfig(BaseModel):
    n_estimators: int
    bootstrap: bool
    random_state: int
    class_weight: Dict[int, float]
    n_jobs: int

class XgbConfig(BaseModel):
    objective: str
    random_state: int
    class_weight: Dict[int, float]
    n_estimators: int
    learning_rate: float
    device: str
    gamma: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    min_child_weight: int
    colsample_bytree: float
    n_jobs: int

class LgbConfig(BaseModel):
    objective: str
    random_state: int
    device: str
    n_estimators: int
    n_jobs: int
    class_weight: Dict[int, float]
    learning_rate: float
    min_child_sample: int

class LogConfig(BaseModel):
    target: str
    used_model: str
    samples_to_train_ratio: float
    to_drop: List[str]
    object_features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    random_state: int
    test_size: float
    vars_with_na: List[str]
    time_transform: str
    use_sampling: bool
    smote: SmoteConfig
    logistic: LogisticRegressionConfig
    random_forest: RandomForestConfig 
    xgb: XgbConfig
    lgb: LgbConfig
    precision_recall_threshold: float

    @validator("to_drop",pre=True, allow_reuse=True)
    def convert_empty_string_to_none(cls, value):
        return [] if value == [''] else value


# Cross validation
class StratifiedKFoldConfig(BaseModel):
    n_splits: int
    shuffle: bool
    random_state: int

    
class RandomForestGridConfig(BaseModel):
    random_forest__criterion: List[str]
    random_forest__n_estimators: List[int]
    random_forest__max_depth: List[Union[None, int]]
    random_forest__min_samples_leaf: List[int]
    random_forest__min_samples_split: List[int]

    @validator("random_forest__max_depth", pre=True, each_item=True, allow_reuse=True)
    def convert_empty_string_to_none(cls, value):
        if value == "":
            return None
        return value
    
class XGboostGridConfig(BaseModel):
    xgboost__learning_rate: List[float]
    xgboost__gamma: List[float]
    xgboost__reg_alpha: List[float]
    xgboost__reg_lambda: List[float]
    xgboost__max_depth: List[int]
    xgboost__min_child_weight: List[int]
    xgboost__subsample: List[float]
    xgboost__colsample_bytree: List[float]



class CVConfig(BaseModel):
    stratifiedkfold: StratifiedKFoldConfig
    random_forest: RandomForestGridConfig
    xgboost: XGboostGridConfig
    
# Mlflow        
class MLflowConfig(BaseModel):
    experiment_name: str
    experiment_tags: Dict[str, str]
    artifact_path: str
    run_name: str

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    log_config: LogConfig
    mlflow_config: MLflowConfig
    cv_config: CVConfig


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
        mlflow_config=MLflowConfig(**parsed_config['mlflow_config'].data),
        cv_config=CVConfig(**parsed_config['cv_config'].data)
    )

    return _config

config = create_and_validate_config()