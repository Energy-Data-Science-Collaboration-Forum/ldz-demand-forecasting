import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import src.prepare_data
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_cwv,
    prepare_gas_demand_diff,
    prepare_cwv_diff,
    add_workday,
    add_christmas_bank_holiday,
    add_weekend_indicator
)


def test_prepare_gas_demand_actuals(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "ApplicableFor": [pd.to_datetime("2022-01-10")] * 5,
            "Value": [1.0] * 5,
            "TYPE": [
                "NTS Volume Offtaken, Industrial Offtake Total",
                "NTS Volume Offtaken, Interconnector Exports Total",
                "NTS Volume Offtaken, LDZ Offtake Total",
                "NTS Volume Offtaken, Powerstations Total",
                "NTS Volume Offtaken, Storage Injection Total",
            ],
        }
    )

    def mock_read_csv(fp, parse_dates):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_gas_demand_actuals("")

    desired_result = pd.DataFrame(
        {
            "INDUSTRIAL": [1.0],
            "INTERCONNECTOR": [1.0],
            "LDZ": [1.0],
            "PS": [1.0],
            "STORAGE": [1.0],
        },
        index=pd.DatetimeIndex([pd.to_datetime("2022-01-10")], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_cwv(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "ApplicableAt": [pd.to_datetime("2022-01-10")] * 13,
            "Value": [1] * 13,
            "LDZ": [
                "SC",
                "NO",
                "NW",
                "NE",
                "EM",
                "WM",
                "WN",
                "WS",
                "EA",
                "NT",
                "SE",
                "SO",
                "SW",
            ],
        }
    )
    mock_data["Value"] = mock_data["Value"].astype(str)
    mock_data["ApplicableFor"] = mock_data["ApplicableAt"]

    def mock_read_csv(fp, parse_dates):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_cwv("")

    desired_result = pd.DataFrame(
        {"CWV": [1.0]},
        index=pd.DatetimeIndex([pd.to_datetime("2022-01-10")], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_gas_demand_diff(monkeypatch):

    mock_data = pd.DataFrame(
        {
            "INDUSTRIAL": [1.0] * 4,
            "INTERCONNECTOR": [1.0] * 4,
            "LDZ": [1.0, 2, 4, 10],
            "PS": [1.0] * 4,
            "STORAGE": [1.0] * 4,
        },
        index=pd.DatetimeIndex(
            pd.date_range("2023-01-20", "2023-01-23"), name="GAS_DAY"
        ),
    )

    def mock_prep(fp):
        return mock_data

    monkeypatch.setattr(src.prepare_data, "prepare_gas_demand_actuals", mock_prep)

    result = prepare_gas_demand_diff(None)
    desired_result = pd.DataFrame(
        {"LDZ_DEMAND_DIFF": [np.NaN, np.NaN, 3, 8]},
        index=pd.DatetimeIndex(
            pd.date_range("2023-01-20", "2023-01-23"), name="GAS_DAY"
        ),
    )
    assert_frame_equal(result, desired_result)


def test_prepare_cwv_diff(monkeypatch):

    mock_data = pd.DataFrame(
        {"CWV": [1.0, 2, 4, 10]},
        index=pd.DatetimeIndex(
            pd.date_range("2023-01-20", "2023-01-23"), name="GAS_DAY"
        ),
    )

    def mock_prep(fp):
        return mock_data

    monkeypatch.setattr(src.prepare_data, "prepare_cwv", mock_prep)

    result = prepare_cwv_diff(None)
    desired_result = pd.DataFrame(
        {"CWV_DIFF": [np.NaN, np.NaN, 3, 8]},
        index=pd.DatetimeIndex(
            pd.date_range("2023-01-20", "2023-01-23"), name="GAS_DAY"
        ),
    )
    assert_frame_equal(result, desired_result)


def test_add_workday():
    mock_data = pd.DataFrame(
        {"One": [1] * 10}, index=pd.date_range("2023-01-30", periods=10, freq="D")
    )

    desired_result = mock_data.copy()
    result = add_workday(mock_data)

    desired_result["WORK_DAY"] = [1] * 5 + [0] * 2 + [1] * 3

    assert_frame_equal(result, desired_result)

def test_add_christmas_bank_holiday():
    mock_data = pd.DataFrame(
        {"One": [1] * 10}, index=pd.date_range("2022-12-24", periods=10, freq="D")
    )

    desired_result = mock_data.copy()
    result = add_christmas_bank_holiday(mock_data)

    desired_result["CHRISTMAS_DAY"] = [0, 1] + [0] * 8
    desired_result["NEW_YEARS_DAY"] = [0] * 8 + [1, 0]
    desired_result["NEW_YEARS_EVE"] = [0] * 7 + [1, 0, 0]
    desired_result["BOXING_DAY"] = [0, 0, 1] + [0] * 7   
        
    assert_frame_equal(result, desired_result)


def test_add_weekend_indicator():
    mock_data = pd.DataFrame(
        {"One": range(7)}, index=pd.date_range("2022-03-21", "2022-03-27")
    )

    result = add_weekend_indicator(mock_data)
    desired_result = pd.DataFrame(
        {"One": range(7), "WEEKEND": [0, 0, 0, 0, 0, 1, 1]},
        index=pd.date_range("2022-03-21", "2022-03-27"),
    )

    assert_frame_equal(result, desired_result, check_dtype=False)