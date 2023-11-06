from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from feature_engine.selection import DropFeatures

from logistic.config.core import config
from logistic.processing import transform_features as pp

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
        ('SMOTE', SMOTE(**config.log_config.smote.model_dump())),
        ('Logistic Regression', LogisticRegression(**config.log_config.logistic.model_dump()))
        
    ]
)
