from model.training import train
from model.training_mlflow import mlflow_train
from model.predict import prediction

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', 
                        help='choose between train, mlflow_train, or predict')
    parser.add_argument('arg2', 
                        nargs='?', 
                        help='for train and mlflow_train please enter train type: train, grid_search, and cross_validation \
                                      for predict enter the pipeline name, i.e. xgboost_output_v0.0.1.pkl. It can also be none')
    args = parser.parse_args()

    if args.arg1 == 'train':
        train_type = args.arg2
        if train_type:
            train(train_type=train_type)
        else:
            train()
    elif args.arg1 == 'mlflow_train':
        train_type = args.arg2
        if train_type:
            mlflow_train(train_type=train_type)
        else:
            mlflow_train()
    elif args.arg1 == 'predict':
        pipeline_file_name = args.arg2
        if pipeline_file_name:
            prediction(pipeline_file_name=pipeline_file_name)
        else:
            prediction()
    else:
        print('Wrong input please see -help')
