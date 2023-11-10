from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from logistic.config.core import config
from logistic.processing import transform_features as pp

models = {'Logistic Regression': LogisticRegression(**dict(config.log_config.logistic)),
          'Random Forest': RandomForestClassifier(**dict(config.log_config.random_forest))}

fill_na = ColumnTransformer(
    transformers=[
        ('fill_na', SimpleImputer(strategy="most_frequent"), config.log_config.vars_with_na)
    ],
    remainder='passthrough'
)

pipe = Pipeline(
    [
        (
            'time_transformation',
            pp.TimeTransformer(variables=config.log_config.time_transform
            ),
        ),
        ('na_values_imputation', fill_na),
        ('scaler', RobustScaler()),
        ('SMOTE', SMOTE(**dict(config.log_config.smote))),
        (config.log_config.used_model, models[config.log_config.used_model])
        
    ]
)
