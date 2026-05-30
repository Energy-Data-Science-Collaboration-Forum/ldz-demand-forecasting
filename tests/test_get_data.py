import pandas as pd
import src.get_data
from src.get_data import get_cwv_from_mipi, get_gas_actuals_from_mipi


def test_get_cwv_from_mipi(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "DUMMY": [1.0, 2.0],
            "PublicationName": [
                "Composite Weather Variable, Actual, LDZ(EA), D+1",
                "Composite Weather Variable, Actual, LDZ(EM), D+1",
            ],
        }
    )

    def mock_get_mipi_data(items, fromdt, todt):
        return mock_data

    monkeypatch.setattr(src.get_data, "get_mipi_data", mock_get_mipi_data)

    def mock_to_csv(self, fp, index):
        return None

    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    get_cwv_from_mipi("", "", "")
    assert True


def test_get_gas_actuals_from_mipi(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "Value": ["1.0", "2.0"],
            "PublicationName": [
                "NTS Volume Offtaken, Industrial Offtake Total",
                "NTS Volume Offtaken, Interconnector Exports Total",
            ],
        }
    )

    def mock_get_mipi_data(items, fromdt, todt):
        return mock_data

    monkeypatch.setattr(src.get_data, "get_mipi_data", mock_get_mipi_data)

    def mock_to_csv(self, fp, index):
        return None

    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    get_gas_actuals_from_mipi("", "", "")
    assert True
