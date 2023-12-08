from model.processing.transform_features import TimeTransformer, NewNAColumn
import pytest
import pandas as pd
import numpy as np

class Test_TimeTransformer:
    def test_time_transformer(self, sample_input_data):
        transformer = TimeTransformer('loctm')
        assert sample_input_data.iloc[1, 1] == 235959

        subject = transformer.fit_transform(sample_input_data)

        actual = [subject.iloc[i, 1] for i in range(len(subject))]
        expected = [0, 1, 0.500006, 0.750009, 0.869443, 0.300374]
        assert all([a == pytest.approx(b, 1e-05) for a, b in zip(actual, expected)])

class Test_NewNAColumn:
    def test_add_one_column(self):
            df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5]})
            transformer = NewNAColumn(['A'])
            transformed_df = transformer.transform(df)
        
            assert 'A_na' in transformed_df.columns

    def test_add_multiple_columns(self):
            df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5]})
            transformer = NewNAColumn(['A', 'B'])
            transformed_df = transformer.transform(df)
    
            assert 'A_na' in transformed_df.columns
            assert 'B_na' in transformed_df.columns

    def test_new_column_value(self):
        # Arrange
        df = pd.DataFrame({'A': [1, 2, np.nan]})
        transformer = NewNAColumn(['A'])
    
        # Act
        transformed_df = transformer.transform(df)
    
        # Assert
        assert transformed_df['A_na'].tolist() == [0, 0, 1]

    def test_no_columns_specified(self):
        df = pd.DataFrame({'A': [1, 2, np.nan]})
        transformer = NewNAColumn([])
        transformed_df = transformer.transform(df)
    
        assert transformed_df.equals(df)