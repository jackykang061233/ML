import pandas as pd
from model.processing.validation import validate_inputs
from model.config.core import config

### INCOMPLETE ###
class Test_Valid_Input:
    def test_valid_input_data_no_missing_values(self):
        # Arrange
        input_data = pd.DataFrame({
            'txkey': ['1', '2', '3'],
            'locdt': [1, 2, 3],
            'loctm': [100000, 200000, 300000],
            'chid': ['A', 'B', 'C'],
            'cano': ['111', '222', '333'],
            'contp': [1, 2, 3],
            'etymd': [1.0, 2.0, 3.0],
            'mchno': ['M1', 'M2', 'M3'],
            'acqic': ['A1', 'A2', 'A3'],
            'mcc': [100.0, 200.0, 300.0],
            'conam': [1000.0, 2000.0, 3000.0],
            'ecfg': [1, 0, 1],
            'insfg': [1, 0, 1],
            'iterm': [1.0, 2.0, 3.0],
            'bnsfg': [1, 0, 1],
            'flam1': [1, 0, 1],
            'stocn': [100.0, 200.0, 300.0],
            'scity': [100.0, 200.0, 300.0],
            'stscd': [1.0, 2.0, 3.0],
            'ovrlt': [1, 0, 1],
            'flbmk': [1, 0, 1],
            'hcefg': [1.0, 2.0, 3.0],
            'csmcu': [100.0, 200.0, 300.0],
            'csmam': [1000, 2000, 3000],
            'flg_3dsmk': [1, 0, 1],
            'label': [0, 1, 0]
        })

        features = [col for col in input_data.columns if col not in config.log_config.to_drop]
        expected_validated_data = input_data[features]
        expected_errors = None

        # Act
        validated_data, errors = validate_inputs(input_data=input_data)

        # Assert
        pd.testing.assert_frame_equal(validated_data, expected_validated_data)
        assert errors == expected_errors
        # Valid input data with some missing values
    def test_valid_input_data_some_missing_values(self):
        # Arrange
        input_data = pd.DataFrame({
            'txkey': ['1', '2', '3'],
            'locdt': [1, None, 3],
            'loctm': [100000, None, 300000],
            'chid': ['A', None, 'C'],
            'cano': ['111', None, '333'],
            'contp': [1, None, 3],
            'etymd': [1.0, None, 3.0],
            'mchno': ['M1', None, 'M3'],
            'acqic': ['A1', None, 'A3'],
            'mcc': [100.0, None, 300.0],
            'conam': [1000.0, None, 3000.0],
            'ecfg': [1, None, 1],
            'insfg': [1, None, 1],
            'iterm': [1.0, None, 3.0],
            'bnsfg': [1, None, 1],
            'flam1': [1, None, 1],
            'stocn': [100.0, None, 300.0],
            'scity': [100.0, None, 300.0],
            'stscd': [1.0, None, 3.0],
            'ovrlt': [1, None, 1],
            'flbmk': [1, None, 1],
            'hcefg': [1.0, None, 3.0],
            'csmcu': [100.0, None, 300.0],
            'csmam': [1000, None, 3000],
            'flg_3dsmk': [1, None, 1],
            'label': [0, None, 0]
        })

        expected_validated_data = pd.DataFrame({
            'txkey': ['1', '3'],
            'locdt': [1, 3],
            'loctm': [100000, 300000],
            'chid': ['A', 'C'],
            'cano': ['111', '333'],
            'contp': [1, 3],
            'etymd': [1.0, 3.0],
            'mchno': ['M1', 'M3'],
            'acqic': ['A1', 'A3'],
            'mcc': [100.0, 300.0],
            'conam': [1000.0, 3000.0],
            'ecfg': [1, 1],
            'insfg': [1, 1],
            'iterm': [1.0, 3.0],
            'bnsfg': [1, 1],
            'flam1': [1, 1],
            'stocn': [100.0, 300.0],
            'scity': [100.0, 300.0],
            'stscd': [1.0, 3.0],
            'ovrlt': [1, 1],
            'flbmk': [1, 1],
            'hcefg': [1.0, 3.0],
            'csmcu': [100.0, 300.0],
            'csmam': [1000, 3000],
            'flg_3dsmk': [1, 1],
            'label': [0, 0]
        })
        expected_errors = None

        # Act
        validated_data, errors = validate_inputs(input_data=input_data)
        print(validated_data)

        # Assert
        pd.testing.assert_frame_equal(validated_data, expected_validated_data)
        assert errors == expected_errors

        # Valid input data with all missing values
    def test_valid_input_data_all_missing_values(self):
        # Arrange
        input_data = pd.DataFrame({
            'txkey': [None, None, None],
            'locdt': [None, None, None],
            'loctm': [None, None, None],
            'chid': [None, None, None],
            'cano': [None, None, None],
            'contp': [None, None, None],
            'etymd': [None, None, None],
            'mchno': [None, None, None],
            'acqic': [None, None, None],
            'mcc': [None, None, None],
            'conam': [None, None, None],
            'ecfg': [None, None, None],
            'insfg': [None, None, None],
            'iterm': [None, None, None],
            'bnsfg': [None, None, None],
            'flam1': [None, None, None],
            'stocn': [None, None, None],
            'scity': [None, None, None],
            'stscd': [None, None, None],
            'ovrlt': [None, None, None],
            'flbmk': [None, None, None],
            'hcefg': [None, None, None],
            'csmcu': [None, None, None],
            'csmam': [None, None, None],
            'flg_3dsmk': [None, None, None],
            'label': [None, None, None]
        })

        expected_validated_data = pd.DataFrame(columns=input_data.columns)
        expected_errors = None

        # Act
        validated_data, errors = validate_inputs(input_data=input_data)

        # Assert
        pd.testing.assert_frame_equal(validated_data, expected_validated_data)
        assert errors == expected_errors

        # Input data with extra fields
    def test_input_data_extra_fields(self):
        # Arrange
        input_data = pd.DataFrame({
            'txkey': ['1', '2', '3'],
            'locdt': [1, 2, 3],
            'loctm': [100000, 200000, 300000],
            'chid': ['A', 'B', 'C'],
            'cano': ['111', '222', '333'],
            'contp': [1, 2, 3],
            'etymd': [1.0, 2.0, 3.0],
            'mchno': ['M1', 'M2', 'M3'],
            'acqic': ['A1', 'A2', 'A3'],
            'mcc': [100.0, 200.0, 300.0],
            'conam': [1000.0, 2000.0, 3000.0],
            'ecfg': [1, 0, 1],
            'insfg': [1, 0, 1],
            'iterm': [1.0, 2.0, 3.0],
            'bnsfg': [1, 0, 1],
            'flam1': [1, 0, 1],
            'stocn': [100.0, 200.0, 300.0],
            'scity': [100.0, 200.0, 300.0],
            'stscd': [1.0, 2.0, 3.0],
            'ovrlt': [1, 0, 1],
            'flbmk': [1, 0, 1],
            'hcefg': [1.0, 2.0, 3.0],
            'csmcu': [100.0, 200.0, 300.0],
            'csmam': [1000, 2000, 3000],
            'flg_3dsmk': ['1', '0', '1'],
            'label': [0, 1, 0],
            'extra_field': ['extra1', 'extra2', 'extra3']
        })

        features = [col for col in input_data.columns if col not in config.log_config.to_drop]
        expected_validated_data = input_data[features]
        expected_validated_data = input_data.drop(columns=['extra_field'])
        expected_errors = None

        # Act
        validated_data, errors = validate_inputs(input_data=input_data)

        # Assert
        pd.testing.assert_frame_equal(validated_data, expected_validated_data)
        assert errors == expected_errors