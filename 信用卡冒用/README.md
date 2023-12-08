# AI-CUP-2023-E.SUN

# Background
This is the code for the competition [AI-CUP-2023-E.SUN](https://tbrain.trendmicro.com.tw/Competitions/Details/31) , which mainly consists of
* data loading
* sklearn pipeline for feature engineering
* training
  * normal training
  * mlflow training
* cross-validation
* grid-search
* prediction 

# File usuage:
* model/: Model training, prediction, feature engineering, config, basically every detail of the model
  * training.py: train the model. There are three train types:
    * train
    * cross-validation
    * grid-search 
  * training_mlflow.py: train the model and log with [mlflow](https://mlflow.org/docs/latest/index.html). There are three train types:
    * train
    * cross-validation
    * grid-search 
  * predict.py: use the saved trained model pipeline make predictions and store in /submissions
  * pipeline.py: the sklearn pipeline of feature engineering and model
  * evaluate.py: evaluate the trained model with 5 metrics: accuracy, precision, recall, f1-score, and auc score.
  * config.yml: all the training details, including where the data are the configuration for the feature engineering, hyperparameters for the model, etc....
* test/: Unit tests for the code
  * WARNING: test_validation and test_prediction still produce errors
* requirements.txt: required packages
* main.py: execute the training process or make prediction, there are two args\
  First argument
  * train \
    Second argument
    * train
    * cross_validation
    * grid_search\
  First argument
  * mlflow_train\
    Second argument
    * train
    * cross-validation
    * grid-search\
  First argument  
  * predict\
    Second argument
    * the name of pipeline, e.g. xgboost_output_v0.0.1.pkl\
   NOTE: Second argument can be ignored, then the default value will be used

# Execution Process:

```
# Install required packages
$ pip install -r requirements.txt 

# Training and inference
$ python main.py train grid_search
```

# 執行流程:

```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python Preprocess/pipeline.py 

# training and inference
$ python main.py
```
