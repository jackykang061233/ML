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
model/: Model training and prediction
train.py: Train the model using preprocessed data.
predict.py: Use the trained model from train.py to make predictions and store the final predictions in final_prediction.csv.
test/: Unit tests for every basic function
test_validation and test_prediction still produce errors.
requirements.txt: Required packages.
main.py: Execute the entire training and prediction process.

# Execution Process:

```
# Install required packages
$ pip install -r requirements.txt 

# Execute data preprocessing
$ python Preprocess/pipeline.py 

# Training and inference
$ python main.py
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
