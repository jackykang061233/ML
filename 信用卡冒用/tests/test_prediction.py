import pytest
from model.predict import prediction, make_prediction
import os
import pandas as pd
import numpy as np

from model import __version__ as _version
from model.config.core import ROOT, config
import model

### UNFINISHED
class Test_Make_Prediction:
        # Given an invalid pipeline file name and a valid predict data, the function should raise an exception.
    def test_invalid_pipeline_and_valid_predict_data(self, mocker):
        # Arrange
        pipeline_file_name = "invalid_pipeline.pkl"
        predict_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        mocker.patch('model.predict.load_save_file', side_effect=Exception("Invalid pipeline file"))

        # Act and Assert
        with pytest.raises(Exception):
            make_prediction(pipeline_file_name=pipeline_file_name, predict_data=predict_data)

        # Given a valid pipeline file name and an empty predict data, the function should raise an exception.
    def test_valid_pipeline_and_empty_predict_data(self, mocker):
        # Arrange
        pipeline_file_name = "valid_pipeline.pkl"
        predict_data = pd.DataFrame()

        # Act and Assert
        with pytest.raises(Exception):
            make_prediction(pipeline_file_name=pipeline_file_name, predict_data=predict_data)

        # Given a valid pipeline file name and a valid predict data, the function should return a dictionary with the predicted values.
    def test_valid_pipeline_and_valid_predict_data(self, mocker):
        # Arrange
        pipeline_file_name = "valid_pipeline.pkl"
        predict_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        # Mock the load_save_file function
        mocker.patch('model.predict.load_save_file')

        # Mock the _pipe object
        _pipe = mocker.Mock()

        # Mock the predict_proba method
        mocker.patch.object(_pipe, 'predict_proba')
        _pipe.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])

        # Mock the make_prediction function
        mocker.patch('model.predict.make_prediction', return_value=[1, 0, 1])
        
        # Call the make_prediction function
        result = model.predict.make_prediction(pipeline_file_name=pipeline_file_name, predict_data=predict_data)

        # Assert that load_save_file was called with the correct arguments
        model.predict.load_save_file.assert_called_once_with(pipeline_file_name)

        # Assert that _pipe.predict_proba was called with the correct arguments
        _pipe.predict_proba.assert_called_once_with(predict_data)

        # Assert the result is correct
        assert result == [1, 0, 1]
class Test_Prediction:
        # Function successfully reads predict_data from file and makes predictions
        # Function successfully reads predict_data from file and makes predictions
    def test_read_predict_data(self, mocker):
        # Mock the pd.read_csv function
        mocker.patch('pandas.read_csv')
        pd.read_csv.return_value = pd.DataFrame({'txkey': [1, 2, 3], 'feature1': [0.5, 0.2, 0.8]})

        # Mock the make_prediction function
        mocker.patch('model.predict.make_prediction')
        model.predict.make_prediction.return_value = [1, 0, 1]

        # Mock the pd.DataFrame function
        mocker.patch('pandas.DataFrame')
        pd.DataFrame.return_value = pd.DataFrame({'txkey': [1, 2, 3], 'pred': [1, 0, 1]})

        # Mock the pd.to_csv function
        mocker.patch('pandas.DataFrame.to_csv')

        # Mock the load_pipeline function
        mocker.patch('model.processing.data_manager.load_pipeline')

        # Mock the joblib.load function
        mocker.patch('joblib.load')

        # Create a mock object for _pipe
        _pipe = mocker.Mock()

        # Mock the predict_proba method
        mocker.patch.object(_pipe, 'predict_proba')
        _pipe.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])

        # Call the prediction function
        prediction('pipeline.pkl')

        # Assert that pd.read_csv was called with the correct arguments
        pd.read_csv.assert_called_once_with(str(ROOT) + config.app_config.predict_path)

        # Assert that make_prediction was called with the correct arguments
        model.predict.make_prediction.assert_called_once_with(pipeline_file_name='pipeline.pkl', predict_data=pd.DataFrame({'txkey': [1, 2, 3], 'feature1': [0.5, 0.2, 0.8]}))

        # Assert that pd.DataFrame was called with the correct arguments
        pd.DataFrame.assert_called_once_with({'txkey': [1, 2, 3], 'pred': [1, 0, 1]})

        # Assert that pd.to_csv was called with the correct arguments
        pd.DataFrame.to_csv.assert_called_once_with(f'{str(ROOT)}/submissions/{config.app_config.pipeline_save_file}{_version}.csv', index=False)
