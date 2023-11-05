from logistic.processing.transform_features import TimeTransformer
import pytest

def test_time_transformer(sample_input_data):
    transformer = TimeTransformer('loctm')
    assert sample_input_data.iloc[1, 1] == 235959

    subject = transformer.fit_transform(sample_input_data)

    actual = [subject.iloc[i, 1] for i in range(len(subject))]
    expected = [0, 1, 0.500006, 0.750009, 0.869443, 0.300374]
    assert all([a == pytest.approx(b, 1e-05) for a, b in zip(actual, expected)])
