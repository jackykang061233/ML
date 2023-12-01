# basic packags
import matplotlib.pyplot as plt

from model.config.core import config, TRAINED_MODEL_DIR
from model.pipeline import pipeline
from model.evaluate import evaluation, cross_validation, grid_search_cv
from model.processing.data_manager import save_pipeline
from model.processing.data_manager import data_prep, custom_val_set, public_as_val
from model.exception import InvalidTrainTypeException
from model import __version__ as _version

def train(train_type='train'):
    X_train, X_test, y_train, y_test = public_as_val()

    print('--------START TRAINING--------')
    print(f'Training size {len(X_train)}')
    print(f'Testing size {len(X_test)}')

    if train_type == 'train':
        pipe = pipeline(X_train.columns)
        model = pipe.fit(X_train, y_train)
    elif train_type == 'grid_search':
        model, params, pipe = grid_search_cv(X_train, y_train)
        model.fit(X_train, y_train)
    elif train_type == 'cross_validation':
        cross_validation(X_train, y_train)
    else:
        raise InvalidTrainTypeException('Invalid! Train type is either train, grid_search, or cross_validation')


    print('--------END TRAINING--------')
    if train_type != 'cross_validation':
        save_pipeline(pipeline_to_save=model)

        pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
        evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)
    if train_type == 'grid_search':
        grid_search_path = str(TRAINED_MODEL_DIR) + '/grid_search/' + f"{config.app_config.pipeline_save_file}.txt"
        with open(grid_search_path, 'w') as f:
            for param_name, param in params.items():
                f.write(f'{param_name}: {param}'+'\n')


# def train():
#     #X_train, X_test, y_train, y_test = custom_val_set()
#     X_train, X_test, y_train, y_test = data_prep()

#     print('--------START TRAINING--------')
#     print(f'Training size {len(X_train)}')
#     print(f'Testing size {len(X_test)}')
#     pipe = pipeline(X_train.columns)
#     model = pipe.fit(X_train, y_train)

#     print('--------END TRAINING--------')
#     save_pipeline(pipeline_to_save=model)

#     pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

# def train_cross_val():
#     X_train, _ , y_train, _ = data_prep()
    
#     print('--------START TRAINING--------')

#     cross_validation(X_train, y_train)
        
#     print('--------END TRAINING--------')
    
# def train_grid_search():
#     #X_train, X_test, y_train, y_test = data_prep()
#     X_train, X_test, y_train, y_test = public_as_val()
    
#     print('--------START TRAINING--------')
#     print(f'Training size {len(X_train)}')
#     print(f'Testing size {len(X_test)}')
#     model, params, pipe = grid_search_cv(X_train, y_train)
#     model.fit(X_train, y_train)
#     print('--------END TRAINING--------')
#     save_pipeline(pipeline_to_save=model)

#     pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     evaluation(pipeline_file_name=pipeline_file_name, test_data=X_test, y_test=y_test)

#     grid_search_path = str(TRAINED_MODEL_DIR) + '/grid_search/' + f"{config.app_config.pipeline_save_file}.txt"
#     with open(grid_search_path, 'w') as f:
#         for param_name, param in params.items():
#             f.write(f'{param_name}: {param}'+'\n')


if __name__ == '__main__':
    train()