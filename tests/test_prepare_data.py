import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from src.prepare_data import prepare_gas_demand_actuals, prepare_cwv


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

    desired_result = pd.Series(
        [1.0],
        name="CWV",
        index=pd.DatetimeIndex([pd.to_datetime("2022-01-10")], name="GAS_DAY"),
    )

    assert_series_equal(result, desired_result)

