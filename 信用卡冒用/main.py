from model.training import train
from model.training_mlflow import mlflow_train
from model.predict import prediction

if __name__ == '__main__':
    train()
    # mlflow_train()
    # prediction(pipeline_file_name=f"{config.app_config.pipeline_save_file}{_version}.pkl")