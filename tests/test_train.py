import pandas as pd
from pandas.testing import assert_series_equal

from src.train import train_ldz_diff


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

    result = train_ldz_diff(target, features)

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
            ]
        ,
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-18", "2022-07-01"), name="GAS_DAY", freq=None
        ),
        name="PREDICTION"
    )

    assert_series_equal(result, desired_result)
