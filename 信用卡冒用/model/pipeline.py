from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2

from model.config.core import config
from model.processing import transform_features as pp


def pipeline(columns):
    models = {'logistic_regression': LogisticRegression(**dict(config.log_config.logistic)),
            'random_forest': RandomForestClassifier(**dict(config.log_config.random_forest)),
            'xgboost': xgb.XGBClassifier(**dict(config.log_config.xgb)),
            'lightgbm': LGBMClassifier(**dict(config.log_config.lgb))}

    # transform object values
    # ordinal OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    # target  TargetEncoder(target_type='binary', random_state=42)
    obj_columns = [(index, c) for index, c in enumerate(columns) if c in config.log_config.categorical_features]
    transform_object = ColumnTransformer(
        transformers=[
            ('transform_obj', TargetEncoder(target_type='binary', random_state=42), list(zip(*obj_columns))[0])
        ],
        remainder='passthrough'
    )
    obj_columns_names = list(list(zip(*obj_columns))[1])
    new_order_columns = [c for c in columns if c not in obj_columns_names]
    new_order_columns = obj_columns_names + new_order_columns

    # fill all na values
    na_columns = [(index, c) for index, c in enumerate(new_order_columns) if c in config.log_config.vars_with_na]
    fill_na = ColumnTransformer(
        transformers=[
            ('fill_na', SimpleImputer(strategy="most_frequent"), list(zip(*na_columns))[0])
        ],
        remainder='passthrough'
    )

    na_columns_names = list(list(zip(*na_columns))[1])
    new_order_columns = [c for c in new_order_columns if c not in na_columns_names]
    new_order_columns = na_columns_names + new_order_columns

    # select k best categorical values
    cat_columns = [(index, c) for index, c in enumerate(new_order_columns) if c in config.log_config.categorical_features]
    # with open('test.txt', 'w') as f:
    #     f.write(" ".join([c for index, c in enumerate(columns) if c in config.log_config.categorical_features and c not in config.log_config.to_drop]))
    feature_selection = ColumnTransformer(
        transformers=[
            ('selected_columns', SelectKBest(chi2, k=len(cat_columns)), list(zip(*cat_columns))[0])
        ],
        remainder='passthrough'
    )

    steps = [
            ('obj_transformation', transform_object),
            ('na_values_imputation', fill_na),
            ('feature_selection', feature_selection),
            ('scaler', RobustScaler()),        
        ]
    if 'loctm' in columns:
        steps.insert(0, ('time_transformation', pp.TimeTransformer(variables=config.log_config.time_transform)))
    
    # if under-or oversampling is used
    if config.log_config.use_sampling:
        steps.append(('SMOTE', SMOTE(**dict(config.log_config.smote))))

    # train model
    steps.append((config.log_config.used_model, models[config.log_config.used_model]))
    pipe = Pipeline(steps)
    return pipe

