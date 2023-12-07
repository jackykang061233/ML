# AI-CUP-2023-E.SUN

# Background
This is the code for the competition [AI-CUP-2023-E.SUN](https://tbrain.trendmicro.com.tw/Competitions/Details/31)

# 基本資訊
* 訓練資料為training.csv(在preprocess時為function feature_transform的input)
* 預測資料為predict.csv(在predict時為function prediction的input)
* 最終預測結果會儲存為final_prediction.csv

# 檔案用途:

* Preprocess/: 資料前處理
  * pipeline.py: 使用sklearn pipeline將原始資料轉換，並將
    * 進行轉換的pipeline儲存為feature_transoform_pipe.joblib
    * 轉換過後的訓練資料(不含label)儲存為transformed_training.csv
    * label儲存為label.csv
* Model/: 模型訓練及預測
  * train.py: load preprocess後的資料進行訓練
  * predict.py: 使用train.py訓練過的模型進行預測，並將最終預測儲存在final_prediction.csv
* config/: 基本configuration設定
  * config.yml: 存放訓練所需的configuration
  * core.py: 讀取config.yml並使用pydantic驗證資料的型態
* requirements.txt: 需要的套件
* main.py: 執行整個訓練流程以及預測的部分

# 執行流程:

```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python Preprocess/pipeline.py 

# training and inference
$ python main.py
```
