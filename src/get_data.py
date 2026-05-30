import logging
import datetime as dt
import pandas as pd
from os import path

from src.mipi import get_mipi_data


logger = logging.getLogger(__name__)

FORMAT = "%Y%m%d_%H%M%S"
GAS_ACTUAL_DATA_ITEMS = [
    "NTS Volume Offtaken, Industrial Offtake Total",
    "NTS Volume Offtaken, Interconnector Exports Total",
    "NTS Volume Offtaken, LDZ Offtake Total",
    "NTS Volume Offtaken, Powerstations Total",
    "NTS Volume Offtaken, Storage Injection Total",
]
CWV_DATA_ITEMS = [
    "Composite Weather Variable, Actual, LDZ(EA), D+1",
    "Composite Weather Variable, Actual, LDZ(EM), D+1",
    "Composite Weather Variable, Actual, LDZ(NE), D+1",
    "Composite Weather Variable, Actual, LDZ(NO), D+1",
    "Composite Weather Variable, Actual, LDZ(NT), D+1",
    "Composite Weather Variable, Actual, LDZ(NW), D+1",
    "Composite Weather Variable, Actual, LDZ(SC), D+1",
    "Composite Weather Variable, Actual, LDZ(SE), D+1",
    "Composite Weather Variable, Actual, LDZ(SO), D+1",
    "Composite Weather Variable, Actual, LDZ(SW), D+1",
    "Composite Weather Variable, Actual, LDZ(WM), D+1",
    "Composite Weather Variable, Actual, LDZ(WN), D+1",
    "Composite Weather Variable, Actual, LDZ(WS), D+1",
]
MIPI_URL = "http://marketinformation.natgrid.co.uk/MIPIws-public/public/publicwebservice.asmx?wsdl"


def get_cwv_from_mipi(output_dir, from_date, to_date):
    """Retrieve CWV data between the given dates for each LDZ (separate column for each LDZ)
    Data is written to the given output directory with a timestamped file name.

    Args:
        output_dir (str): directory path to write the output to
        from_date (str): Lower bound for the applicable date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the applicable date of the dataset, yyyy-mm-dd format
    """
    cwvs_list = []
    # Have to get the data for each LDZ separately as the requested data is more than the allowed threshold
    for item in CWV_DATA_ITEMS:
        data = get_mipi_data([item], from_date, to_date)
        if len(data) > 0:
            cwvs_list.append(data)
    
    if len(cwvs_list) > 0:
        cwvs = pd.concat(cwvs_list)
        cwvs["LDZ"] = cwvs["PublicationName"].str.slice(-8, -6)
        df = cwvs.drop(columns="PublicationName")
        df.to_csv(
            path.join(output_dir, f"cwv_data_{dt.datetime.now().strftime(FORMAT)}.csv"),
            index=False,
        )
    else:
        logger.warning("No CWV Data Available")


def get_gas_actuals_from_mipi(output_dir, from_date, to_date):
    """Retrieve actual gas demand data at an aggregated level (see GAS_ACTUAL_DATA_ITEMS)
    between the given dates. Data is written to the given output directory with a timestamped file name.

    Args:
        output_dir (str): directory path to write the output to
        from_date (str): Lower bound for the applicable date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the applicable date of the dataset, yyyy-mm-dd format
    """
    actual_data = get_mipi_data(GAS_ACTUAL_DATA_ITEMS, from_date, to_date)
    if len(actual_data) > 0:
        df = actual_data.rename(columns={"PublicationName": "TYPE"})
        df["Value"] = pd.to_numeric(df["Value"])
        df.to_csv(
            path.join(
                output_dir, f"gas_actuals_{dt.datetime.now().strftime(FORMAT)}.csv"
            ),
            index=False,
        )
    else:
        logger.warning("No Actuals Data Returned")


if __name__ == "__main__":
    data_dir = "data"
    get_cwv_from_mipi(data_dir, "2019-01-01", "2022-11-01")
    get_gas_actuals_from_mipi(data_dir, "2019-01-01", "2022-11-01")
