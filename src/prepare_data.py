import pandas as pd
from workalendar.europe import UnitedKingdom

# weights taken from doc/Gas_Demand_Forecasting_Methodology_Nov2016.pdf
CWV_LDZ_WEIGHTS = pd.DataFrame(
    {
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
        "Weight": [9.3, 5.5, 12.1, 6.4, 10.1, 8.9, 1.1, 3.4, 8.3, 11.4, 10.6, 7.1, 5.8],
    }
)


def prepare_gas_features(file_paths):
    """Gather all the features necessary to predict LDZ gas demand

    Args:
        file_paths (dict): A dictionary of feature names and file paths

    Returns:
        pandas DataFrame: A DataFrame with feature data in the columns and Gas Day on the index
    """
    features = []

    features.append(prepare_cwv(file_paths["CWV"]))

    features.append(prepare_gas_demand_diff(file_paths["GAS_DEMAND"]))

    features.append(prepare_cwv_diff(file_paths["CWV"]))

    features = pd.concat(features, axis=1)

    features = add_workday(features)

    features = add_christmas_bank_holiday(features)

    return features


def add_christmas_bank_holiday(input_data):
    """Adds 4 indicator columns for the bank holidays around Christmas

    Args:
        input_data (pandas DataFrame): A DataFrame with dates on the index

    Returns:
        pandas DataFrame: A DataFrame with an additional CHRISTMAS_DAY, NEW_YEARS_DAY, NEW_YEARS_EVE 
        and BOXING_DAY column. Values are 0 (= no bank holiday) and 1 (= bank holiday) values
    """
    result = input_data.copy()

    result["M"] = result.index.month
    result["D"] = result.index.day
    result["CHRISTMAS_DAY"] = result["NEW_YEARS_DAY"] = result["NEW_YEARS_EVE"] = result[
        "BOXING_DAY"
    ] = 0
    result.loc[(result["M"] == 12) & (result["D"] == 25), "CHRISTMAS_DAY"] = 1
    result.loc[(result["M"] == 1) & (result["D"] == 1), "NEW_YEARS_DAY"] = 1
    result.loc[(result["M"] == 12) & (result["D"] == 31), "NEW_YEARS_EVE"] = 1
    result.loc[(result["M"] == 12) & (result["D"] == 26), "BOXING_DAY"] = 1

    result = result.drop(columns=["M", "D"])
    
    return result


def add_workday(input_data):
    """Adds a workday indicator to the given input data

    Args:
        input_data (pandas DataFrame): A DataFrame with dates on the index

    Returns:
        pandas DataFrame: A DataFrame with an additional WORK_DAY column of 0 (= non working day) and 1 (= working day) values
    """

    result = input_data.copy()

    cal = UnitedKingdom()
    result["WORK_DAY"] = result.index.to_series().apply(lambda x: 1 if cal.is_working_day(x) else 0)

    return result


def prepare_cwv(file_path):
    """Read the CWV from the given file path and apply the necessary processing

    Args:
        file_path (str): The full file path to the CWV data

    Returns:
        pandas DataFrame: A dataframe with CWV data and Gas Day on the index
    """
    result = pd.read_csv(file_path, parse_dates=["ApplicableFor"])
    result["Value"] = pd.to_numeric(result["Value"])
    result = result.sort_values("ApplicableAt", ascending=False)

    result = (
        result[["ApplicableFor", "Value", "LDZ"]]
        .drop_duplicates(["ApplicableFor", "LDZ"])
        .rename(columns={"ApplicableFor": "GAS_DAY"})
        .assign(
            GAS_DAY=lambda df: df["GAS_DAY"].dt.tz_localize(None)
        )  # remove the timezone awareness
        .merge(CWV_LDZ_WEIGHTS, on="LDZ")
        .assign(Value=lambda df: df["Value"] * df["Weight"] / 100)
        .pivot(index="GAS_DAY", columns="LDZ", values="Value")
        .sum(axis=1)
    )

    result.name = "CWV"

    return result.to_frame()


def prepare_gas_demand_actuals(file_path):
    """Read the gas demand actuals from the given file path and apply the necessary processing.
    The data is reported separately for Interconnectors, Powerstations, Industrials, Storage and LDZ.

    Args:
        file_path (str): The full file path to the gas demand actuals data

    Returns:
        pandas DataFrame: A dataframe with gas demand actuals data and Gas Day on the index
    """
    result = pd.read_csv(file_path, parse_dates=["ApplicableFor"])

    result = result.rename(columns={"ApplicableFor": "GAS_DAY"})
    result = result[["GAS_DAY", "TYPE", "Value"]]
    result["TYPE"] = (
        result["TYPE"]
        .str.replace("NTS Volume Offtaken, ", "")
        .str.replace(" Total", "")
        .str.replace("Industrial Offtake", "Industrial")
        .str.replace("Powerstations", "Powerstation")
        .str.upper()
    )
    result["GAS_DAY"] = result["GAS_DAY"].dt.tz_localize(None)

    demand = result.pivot(index="GAS_DAY", columns="TYPE", values="Value").rename(
        {
            "INTERCONNECTOR EXPORTS": "INTERCONNECTOR",
            "STORAGE INJECTION": "STORAGE",
            "POWERSTATION": "PS",
            "LDZ OFFTAKE": "LDZ",
        },
        axis=1,
    )

    demand.columns.name = None

    demand = demand.fillna(0).sort_index(ascending=True)

    return demand


def prepare_gas_demand_diff(file_path):
    demand = prepare_gas_demand_actuals(file_path)

    demand["LDZ_DEMAND_DIFF"] = demand["LDZ"] - demand["LDZ"].shift(2)

    return demand[["LDZ_DEMAND_DIFF"]]


def prepare_cwv_diff(file_path):
    cwv = prepare_cwv(file_path)

    cwv["CWV_DIFF"] = cwv["CWV"] - cwv["CWV"].shift(2)

    return cwv[["CWV_DIFF"]]
