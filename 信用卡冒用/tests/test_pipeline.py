from model.pipeline import pipeline
from imblearn.pipeline import Pipeline
from model.processing import transform_features as pp
from model.config.core import config

import pandas as pd

class Test_Pipeline:
    def read_df(file_path):
        df = pd.read_csv(file_path)
        # Do something with the DataFrame
        return df
    def test_return_pipeline_object(self):
        columns = ['col1', 'col2', 'col3']
        result = pipeline(columns)
        assert isinstance(result, Pipeline)

    def test_pipeline_steps_order(self):
        columns = ['loctm', 'mcc', 'cano', 'scity']
        pipe = pipeline(columns)
        if config.log_config.use_sampling:
            expected_steps = ['time_transformation',
                              'add_na_column',
                              'obj_transformation',
                              'na_values_imputation',
                              'scaler',
                              'SMOTE',
                              config.log_config.used_model]
        else:
             expected_steps = ['time_transformation',
                               'add_na_column',
                               'obj_transformation',
                               'na_values_imputation',
                               'scaler',
                               config.log_config.used_model]
        assert list(pipe.named_steps.keys()) == expected_steps

    def test_pipeline_with_time_transformation(self):
        columns = ['col1', 'col2', 'col3', config.log_config.time_transform]
        pipe = pipeline(columns)
        assert pipe.steps[0][0] == 'time_transformation'
    
    def test_pipeline_with_add_na_column(self):
        columns = ['col1', 'col2', 'col3', config.log_config.add_na_column[0]]
        pipe = pipeline(columns)
        assert pipe.steps[0][0] == 'add_na_column'

    def test_pipeline_with_obj_transformation(self):
        columns = ['col1', 'col2', 'col3', config.log_config.object_features[0]]
        pipe = pipeline(columns)
        assert pipe.steps[0][0] == 'obj_transformation'
   
    def test_pipeline_with_na_values_imputation(self):
        columns = ['col1', 'col2', 'col3', 'scity']
        pipe = pipeline(columns)
        assert pipe.steps[0][0] == 'na_values_imputation'

    def test_pipeline_with_only_scaler(self):
        columns = ['col1', 'col2', 'col3']
        pipe = pipeline(columns)
        assert pipe.steps[0][0] == 'scaler'

    def test_pipeline_except_smote_model(self, sample_train_data):
        columns = ['loctm', 'mcc', 'cano', 'scity']
        X = sample_train_data
        y = sample_train_data['label']
        X = X[columns]
        
        if config.log_config.use_sampling:
            pipe = pipeline(columns)[:-2]
        else:
            pipe = pipeline(columns)[:-1]
        X_new = pipe.fit_transform(X, y)

        assert len(X_new[0]) == 5
        

    
    