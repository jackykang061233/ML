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
    """
    Create a scikit-learn pipeline for preprocessing and modeling based on configuration settings.
    There are maxmimal seven steps (could be less depending on the training condition)
    - 'time_transformation': transfomr loctm
    - 'add_na_column': add extra binary column for columns with na values (config.log_config.add_na_column)
    - 'obj_transformation': transform object values with target encoding (config.log_config.object_features)
    - 'na_values_imputation': fill nan with most frequent values (config.log_config.vars_with_na)
    - 'scaler': Robust Scaler to scale all the data
    - 'SMOTE': if Smote is used
    - Model: used model. Current: 1. logistic regression 2. random forest 3. xgboost 4. lightgbm

    Parameters:
    - columns (list): List of column names of training data

    Returns:
    Pipeline: A scikit-learn pipeline containing various preprocessing and modeling steps.
    """

    models = {'logistic_regression': LogisticRegression(**dict(config.log_config.logistic)),
            'random_forest': RandomForestClassifier(**dict(config.log_config.random_forest)),
            'xgboost': xgb.XGBClassifier(**dict(config.log_config.xgb)),
            'lightgbm': LGBMClassifier(**dict(config.log_config.lgb))}
    
    steps = []
    new_order_columns = columns

    if config.log_config.time_transform in columns:
        steps.append(('time_transformation', pp.TimeTransformer(variables=config.log_config.time_transform)))
    
    common_add_na_col = list(set(config.log_config.add_na_column).intersection(set(columns)))
    if common_add_na_col:
        steps.append(('add_na_column', pp.NewNAColumn(col=common_add_na_col)))

    ### TRANSFORM OBJECT VALUE ###
    obj_columns = [(index, c) for index, c in enumerate(columns) if c in config.log_config.object_features]
    if obj_columns:
        transform_object = ColumnTransformer(
            transformers=[
                ('transform_obj', TargetEncoder(target_type='binary', random_state=42), list(zip(*obj_columns))[0])
            ],
            remainder='passthrough'
        )
        steps.append(('obj_transformation', transform_object))
        obj_columns_names = list(list(zip(*obj_columns))[1])
        new_order_columns = [c for c in columns if c not in obj_columns_names]
        new_order_columns = obj_columns_names + new_order_columns


    ### FILL NA VALUE
    na_columns = [(index, c) for index, c in enumerate(new_order_columns) if c in config.log_config.vars_with_na]
    if na_columns:
        fill_na = ColumnTransformer(
            transformers=[
                ('fill_na', SimpleImputer(strategy="most_frequent"), list(zip(*na_columns))[0])
            ],
            remainder='passthrough'
        )
        steps.append(('na_values_imputation', fill_na))
        na_columns_names = list(list(zip(*na_columns))[1])
        new_order_columns = [c for c in new_order_columns if c not in na_columns_names]
        new_order_columns = na_columns_names + new_order_columns

    # select k best categorical values
    # cat_columns = [(index, c) for index, c in enumerate(new_order_columns) if c in config.log_config.categorical_features]
    # # with open('test.txt', 'w') as f:
    # #     f.write(" ".join([c for index, c in enumerate(columns) if c in config.log_config.categorical_features and c not in config.log_config.to_drop]))
    # feature_selection = ColumnTransformer(
    #     transformers=[
    #         ('selected_columns', SelectKBest(chi2, k=len(cat_columns)), list(zip(*cat_columns))[0])
    #     ],
    #     remainder='passthrough'
    # )
    # cat_columns_names = list(list(zip(*cat_columns))[1])
    # new_order_columns = [c for c in new_order_columns if c not in cat_columns_names]
    # new_order_columns = cat_columns_names + new_order_columns

    steps.append( ('scaler', RobustScaler()))
    
    # if under-or oversampling is used
    if config.log_config.use_sampling:
        steps.append(('SMOTE', SMOTE(**dict(config.log_config.smote))))

    # train model
    steps.append((config.log_config.used_model, models[config.log_config.used_model]))
    pipe = Pipeline(steps)
    
    return pipe

