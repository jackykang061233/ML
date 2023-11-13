from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from model.config.core import config
from model.processing import transform_features as pp

models = {'logistic_regression': LogisticRegression(**dict(config.log_config.logistic)),
          'random_forest': RandomForestClassifier(**dict(config.log_config.random_forest)),
          'xgboost': xgb.XGBClassifier(**dict(config.log_config.xgb))}

fill_na = ColumnTransformer(
    transformers=[
        ('fill_na', SimpleImputer(strategy="most_frequent"), config.log_config.vars_with_na)
    ],
    remainder='passthrough'
)

steps = [
        (
            'time_transformation',
            pp.TimeTransformer(variables=config.log_config.time_transform
            ),
        ),
        ('na_values_imputation', fill_na),
        ('scaler', RobustScaler()),        
    ]
if config.log_config.use_sampling:
    steps.append(('SMOTE', SMOTE(**dict(config.log_config.smote))))
    
steps.append((config.log_config.used_model, models[config.log_config.used_model]))
pipe = Pipeline(steps)

