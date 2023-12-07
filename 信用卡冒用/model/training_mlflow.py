# basic packags
import numpy as np

from model.config.core import config
from model.pipeline import pipeline
from model.evaluate import grid_search_cv, cross_validation
from model.processing.data_manager import save_pipeline
from model.exception import InvalidTrainTypeException

# mlflow
from mlflow.models.signature import infer_signature
import mlflow

from utils import accuracy, precision, recall, f1, auc
from model.processing.data_manager import data_prep, custom_test_set, public_as_test

models = {'logistic_regression': dict(config.log_config.logistic),
          'random_forest': dict(config.log_config.random_forest),
          'xgboost': dict(config.log_config.xgb),
          'lightgbm': dict(config.log_config.lgb)}


def mlflow_train(train_type='train', models=models):
    """
    Train a machine learning model using MLflow for experiment tracking and model versioning.

    Parameters:
    - train_type (str, optional): The type of training to perform. Options are 'train', 'grid_search', or 'cross_validation'. Default is 'train'.
    - models (dict, optional): A dictionary containing the config of the models.

    Raises:
    - InvalidTrainTypeException: Raised if an invalid train type is provided.

    Examples:
    # train
    mlflow_train()

    # grid search
    mlflow_train(train_type='grid_search')

    # cross-validation
    mlflow_train(train_type='cross_validation')
    ```
    """
     # load data
    X_train, X_test, y_train, y_test = data_prep()

    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')

    experiment_name = config.mlflow_config.experiment_name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, tags=config.mlflow_config.experiment_tags)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=config.mlflow_config.run_name) as run:
        if train_type == 'train':
            pipe = pipeline(X_train.columns)
            model = pipe.fit(X_train, y_train)
        elif train_type == 'grid_search':
            model, params, pipe = grid_search_cv(X_train, y_train)
            model = model.fit(X_train, y_train)
        elif train_type == 'cross_validation':
            accuracy_lst, precision_lst, recall_lst, f1_lst, auc_lst = cross_validation(X_train, y_train)

            accuracy_score = np.mean(accuracy_lst)
            precision_score = np.mean(precision_lst)
            recall_score = np.mean(recall_lst)
            f1_score = np.mean(f1_lst)
            auc_score = np.mean(auc_lst)

            mlflow.log_metrics({
            'accuracy': accuracy_score, 
            'precision': precision_score, 
            'recall': recall_score, 
            'f1': f1_score, 
            'auc': auc_score
            })
            mlflow.log_params({'cv detail': dict(config.cv_config.stratifiedkfold)})
        else:
            raise InvalidTrainTypeException('Invalid! Train type is either train, grid_search, or cross_validation')

        # get selected features' names
        if train_type != 'cross_validation':
            # if train_type == 'grid_search':
            #     select_k_best = model.named_steps['feature_selection'].named_transformers_['selected_columns']
            # else:
            #     select_k_best = pipe.named_steps['feature_selection'].named_transformers_['selected_columns']
            # cols_idxs = select_k_best.get_support(indices=True).tolist()
            # used_categprical_features = [col for col in config.log_config.categorical_features if col not in config.log_config.to_drop]
            # selected_columns = [col for index, col in enumerate(used_categprical_features) if index in cols_idxs]
            # used_numerical_features = [col for col in config.log_config.numeric_features if col not in config.log_config.to_drop]
            # features = selected_columns+used_numerical_features

            ### log used features and pipeline details ###
            features = X_train.columns
            mlflow.log_params({'selected_features': features})
            mlflow.log_params({'pipe details': pipe.named_steps})

            # predictions = model.predict_proba(X_val)
            # predictions = np.where(predictions[:, 1]>=config.log_config.precision_recall_threshold, 1, 0)
            # accuracy_score = accuracy(y_val, predictions)
            # precision_score = precision(y_val, predictions)
            # recall_score = recall(y_val, predictions)
            # f1_score = f1(y_val, predictions)
            # auc_score = auc(y_val, predictions)

            # mlflow.log_metrics({
            #     'val_accuracy': accuracy_score, 
            #     'val_precision': precision_score, 
            #     'val_recall': recall_score, 
            #     'val_f1': f1_score, 
            #     'val_auc': auc_score
            # })

            ### log metric ###
            predictions = model.predict_proba(X_test)
            predictions = np.where(predictions[:, 1]>=config.log_config.precision_recall_threshold, 1, 0)

            accuracy_score = accuracy(y_test, predictions)
            precision_score = precision(y_test, predictions)
            recall_score = recall(y_test, predictions)
            f1_score = f1(y_test, predictions)
            auc_score = auc(y_test, predictions)

            mlflow.log_metrics({
                'test_accuracy': accuracy_score, 
                'test_precision': precision_score, 
                'test_recall': recall_score, 
                'test_f1': f1_score, 
                'test_auc': auc_score
            })

            ### log feature importance ###
            if config.log_config.used_model == 'logistic_regression':
                importances = model.named_steps[config.log_config.used_model].coef_[0]
            else:
                importances = model.named_steps[config.log_config.used_model].feature_importances_
            feature_importance = sorted([(feature, importance) for feature, importance in zip(features, importances)], key=lambda x: x[1], reverse=True)
            mlflow.log_params({'feature importance': feature_importance})
            save_pipeline(pipeline_to_save=model)
            
        
        ### over sample ###
        if config.log_config.use_sampling:
            mlflow.log_params({'smote': dict(config.log_config.smote)})
            
        ### model's parameters ###
        if train_type == 'grid_search':
            mlflow.log_params({config.log_config.used_model: params})
        else:
            mlflow.log_params({config.log_config.used_model: models[config.log_config.used_model]})
        
        ### f1-score threshold ###
        mlflow.log_params({'threshold': config.log_config.precision_recall_threshold})
        
        ### save model ###
        if train_type != 'cross_validation':
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(model, signature=signature, artifact_path=config.mlflow_config.artifact_path)                         
        
    print('--------END TRAINING--------')

