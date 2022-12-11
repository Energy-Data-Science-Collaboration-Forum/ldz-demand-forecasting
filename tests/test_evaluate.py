import pandas as pd
from pandas.testing import assert_frame_equal
from src.evaluate import evaluate_models


def test_evaluate_models():
    mock_predictions = pd.Series(
        [1, 2, 3, 4, 5, 6], index=pd.date_range("2022-08-01", "2022-08-06")
    )
    actuals = pd.DataFrame(
        {"LDZ": [1] * 6}, index=pd.date_range("2022-08-01", "2022-08-06")
    )

    result = evaluate_models(mock_predictions, actuals)
    desired_result = pd.DataFrame(
        {"MODEL": ["LDZ"], "MAE": [2.5], "MAPE": [2.5]}
    )
    assert_frame_equal(result, desired_result)
