import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

import src.train
from src.train import (
    train_ldz_diff,
    get_ldz_match_predictions,
    add_average_demand_by_cwv,
    add_average_demand_by_month_day,
    train_glm,
)


def test_train_glm():
    target = pd.DataFrame(
        {
            "LDZ": [
                52.16314,
                53.94938,
                61.38836,
                59.02986,
                53.926,
                49.71774,
                47.03101,
                49.43077,
                53.47305,
                54.79746,
                54.57056,
                51.4812,
                51.66534,
                52.27608,
                46.27135,
                48.5113,
                56.47527,
                57.09211,
                61.32952,
                63.85277,
                58.93696,
            ]
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )
    features = pd.DataFrame(
        {
            "CWV": [
                9.01,
                3.91,
                6.54,
                6.0,
                15.09,
                4.73,
                8.65,
                5.96,
                11.03,
                3.14,
                12.26,
                7.03,
                11.62,
                6.24,
                7.13,
                4.16,
                9.14,
                9.35,
                4.16,
                9.14,
                9.35,
            ],
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )

    _, result = train_glm(target, features)
    desired_result = pd.Series(
        [
            53.63526375,
            53.37458276,
            53.68013042,
            53.62958443,
            53.79826036,
            52.55993352,
            52.57122378,
            52.29219321,
            52.55993352,
            52.57122378,
        ],
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-22", "2022-07-01"), name="GAS_DAY"
        ),
        name="GLM_CWV",
    )
    assert_series_equal(result, desired_result)


def test_train_ldz_diff():
    target = pd.DataFrame(
        {
            "LDZ": [
                52.16314,
                53.94938,
                61.38836,
                59.02986,
                53.926,
                49.71774,
                47.03101,
                49.43077,
                53.47305,
                54.79746,
                54.57056,
                51.4812,
                51.66534,
                52.27608,
                46.27135,
                48.5113,
                56.47527,
                57.09211,
                61.32952,
                63.85277,
                58.93696,
            ]
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )
    features = pd.DataFrame(
        {
            "LDZ_DEMAND_DIFF": [
                -7.67392,
                -0.82155,
                9.22522,
                5.08048,
                -7.46236,
                -9.31212,
                -6.89499,
                -0.28697,
                6.44204,
                5.36669,
                1.09751,
                -3.31626,
                -2.90522,
                0.79488,
                -5.39399,
                -3.76478,
                10.20392,
                8.58081,
                4.85425,
                6.76066,
                -2.39256,
            ],
            "CWV_DIFF": [
                0.29562,
                -0.28646,
                -0.31025,
                0.2003,
                0.68959,
                0.62967,
                0.2468,
                -0.44743,
                -0.78035,
                -0.10498,
                0.60176,
                0.59314,
                0.26335,
                -0.11795,
                -0.48458,
                -0.43257,
                -0.36436,
                -0.18933,
                0.14992,
                -0.03607,
                -0.17091,
            ],
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )

    _, result = train_ldz_diff(target, features)

    desired_result = pd.Series(
        [
            42.89517882,
            38.87875191,
            39.94676525,
            42.65723483,
            42.65128865,
            41.09468538,
            36.67572682,
            51.63101586,
            52.20381467,
            46.16108121,
            48.36261714,
            56.28754182,
            56.86735549,
            61.06754255,
        ],
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-18", "2022-07-01"), name="GAS_DAY", freq=None
        ),
        name="PROPHET_DIFF_DEMAND",
    )

    assert_series_equal(result, desired_result)


def test_get_ldz_match_predictions_basic():

    dates = pd.date_range("2023-01-29", periods=40, freq="D")

    mock_target = pd.DataFrame(
        {
            "LDZ": range(1, len(dates) + 1),
        },
        index=dates,
    )

    mock_features = pd.DataFrame(
        {
            "CWV": np.linspace(1, 10, num=len(dates)),
            "WORK_DAY": [True] * len(dates),
            "CHRISTMAS_DAY": [False] * len(dates),
            "NEW_YEARS_DAY": [False] * len(dates),
            "NEW_YEARS_EVE": [False] * len(dates),
            "BOXING_DAY": [False] * len(dates),
        },
        index=dates,
    )

    result = get_ldz_match_predictions(mock_target, mock_features)

    desired_result = pd.DataFrame(
        {"LDZ_MATCHED": [30.0] * 7 + [37.0] * 3},
        index=pd.date_range("2023-02-28", periods=10, freq="D"),
    )

    assert_frame_equal(result, desired_result)


def test_get_ldz_match_predictions_with_averages(monkeypatch):

    dates = pd.date_range("2023-01-29", periods=40, freq="D")

    mock_target = pd.DataFrame(
        {
            "LDZ": range(1, len(dates) + 1),
        },
        index=dates,
    )

    mock_features = pd.DataFrame(
        {
            "CWV": np.linspace(1, 10, num=len(dates)),
            "WORK_DAY": [True] * len(dates),
            "CHRISTMAS_DAY": [False] * len(dates),
            "NEW_YEARS_DAY": [False] * len(dates),
            "NEW_YEARS_EVE": [False] * len(dates),
            "BOXING_DAY": [False] * len(dates),
        },
        index=dates,
    )

    mock_features.iloc[-1, 2] = True
    mock_features.iloc[-2, 3] = True
    mock_features.iloc[-3, 4] = True
    mock_features.iloc[-4, 5] = True

    def mock_average_demand_cwv(testd, traind):
        result = testd.copy()
        result["AVERAGE_DEMAND_CWV"] = 1
        return result

    monkeypatch.setattr(src.train, "add_average_demand_by_cwv", mock_average_demand_cwv)

    def mock_average_demand_md(testd, traind):
        result = testd.copy()
        result["AVERAGE_DEMAND_MONTH_DAY"] = 2
        return result

    monkeypatch.setattr(
        src.train, "add_average_demand_by_month_day", mock_average_demand_md
    )

    result = get_ldz_match_predictions(mock_target, mock_features)

    desired_result = pd.DataFrame(
        {"LDZ_MATCHED": [1] * 6 + [2] * 4},
        index=pd.date_range("2023-02-28", periods=10, freq="D"),
    )

    assert_frame_equal(result, desired_result)


def test_add_average_demand_by_month_day():
    mock_data = pd.DataFrame(
        {"MONTH": [1] * 3 + [2] * 4, "DAY": [2] * 3 + [4] * 4, "LDZ": range(1, 8)}
    )

    mock_test_data = pd.DataFrame(
        {"MONTH": [1, 1, 2], "DAY": [2, 3, 4]}, index=[3, 2, 1]
    )

    desired_result = mock_test_data.copy()

    result = add_average_demand_by_month_day(mock_test_data, mock_data)

    desired_result["AVERAGE_DEMAND_MONTH_DAY"] = [2, np.NaN, 5.5]

    assert_frame_equal(result, desired_result)


def test_add_average_demand_by_cwv():
    mock_data = pd.DataFrame(
        {
            "CWV_rounded": [1.2, 1.2, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.8, 1.8],
            "WORK_DAY": [True] * 2 + [False] * 8,
            "CHRISTMAS_DAY": [False] * 2 + [True] * 2 + [False] * 6,
            "NEW_YEARS_DAY": [False] * 4 + [True] * 2 + [False] * 4,
            "NEW_YEARS_EVE": [False] * 6 + [True] * 2 + [False] * 2,
            "BOXING_DAY": [False] * 8 + [True] * 2,
            "LDZ": range(1, 11),
        }
    )

    mock_test_data = pd.DataFrame(
        {
            "CWV_rounded": [1.2, 1.4, 1.5, 1.6, 1.8, 2.0],
            "WORK_DAY": [True] + [False] * 5,
            "CHRISTMAS_DAY": [False] + [True] + [False] * 4,
            "NEW_YEARS_DAY": [False] * 2 + [True] + [False] * 3,
            "NEW_YEARS_EVE": [False] * 3 + [True] + [False] * 2,
            "BOXING_DAY": [False] * 4 + [True] + [False],
        },
        index=[6, 5, 4, 3, 2, 1],
    )

    desired_result = mock_test_data.copy()

    result = add_average_demand_by_cwv(mock_test_data, mock_data)

    desired_result["AVERAGE_DEMAND_CWV"] = [1.5, 3.5, 5.5, 7.5, 9.5, np.NaN]
    assert_frame_equal(result, desired_result)
