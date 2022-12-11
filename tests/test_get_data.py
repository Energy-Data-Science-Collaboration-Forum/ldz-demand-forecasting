import logging
import pandas as pd
from pandas.testing import assert_frame_equal
import src.get_data
from src.get_data import get_cwv_from_mipi, get_gas_actuals_from_mipi, get_mipi_data


def test_get_mipi_data_no_data(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    class MockService:
        def GetPublicationDataWM(self, body):
            return None

    class MockClient:
        service = MockService()

        def __init__(self, url, transport) -> None:
            pass

    monkeypatch.setattr(src.get_data, "Client", MockClient)

    result = get_mipi_data(["One"], "2022-01-01", "2022-01-02")
    assert result == []

    assert len(caplog.messages) == 2
    assert (
        caplog.messages[0]
        == "MIPI LDZ Actual : Gathering One data, from 2022-01-01 to 2022-01-02"
    )
    assert caplog.messages[1] == "No Data for: One"


def test_get_mipi_data(monkeypatch):
    mock_data = [{"One": 1.0, "Two": 2.0}, {"One": 3.0, "Two": 4.0}]

    class MockResponse:
        PublicationObjectData = {"CLSPublicationObjectDataBE": mock_data}

    class MockService:
        def GetPublicationDataWM(self, body):
            return [MockResponse()]

    class MockClient:
        service = MockService()

        def __init__(self, url, transport) -> None:
            pass

    monkeypatch.setattr(src.get_data, "Client", MockClient)

    result = get_mipi_data(["One"], "2022-01-01", "2022-01-02")
    desired_result = pd.DataFrame(
        {"One": [1.0, 3.0], "Two": [2.0, 4.0], "DATA_ITEM": ["One", "One"]}
    )
    assert len(result) == 1  # only 1 data item
    assert_frame_equal(result[0], desired_result)


def test_get_cwv_from_mipi(monkeypatch):
    mock_data = [
        pd.DataFrame(
            {
                "DUMMY": [1.0, 2.0],
                "DATA_ITEM": [
                    "Composite Weather Variable, Actual, LDZ(EA), D+1",
                    "Composite Weather Variable, Actual, LDZ(EM), D+1",
                ],
            }
        )
    ]

    def mock_get_mipi_data(items, fromdt, todt):
        return mock_data

    monkeypatch.setattr(src.get_data, "get_mipi_data", mock_get_mipi_data)

    def mock_to_csv(self, fp, index):
        return None

    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    get_cwv_from_mipi("", "", "")
    assert True


def test_get_gas_actuals_from_mipi(monkeypatch):
    mock_data = [
        pd.DataFrame(
            {
                "Value": ["1.0", "2.0"],
                "DATA_ITEM": [
                    "NTS Volume Offtaken, Industrial Offtake Total",
                    "NTS Volume Offtaken, Interconnector Exports Total",
                ],
            }
        )
    ]

    def mock_get_mipi_data(items, fromdt, todt):
        return mock_data

    monkeypatch.setattr(src.get_data, "get_mipi_data", mock_get_mipi_data)

    def mock_to_csv(self, fp, index):
        return None

    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    get_gas_actuals_from_mipi("", "", "")
    assert True

