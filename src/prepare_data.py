import pandas as pd

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

    features = pd.concat(features, axis=1)

    return features


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

    return result


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
