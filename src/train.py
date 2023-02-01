import warnings
import logging
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet

from src.utils import suppress_stdout_stderr

logger = logging.getLogger(__name__)


def train(target, features):
    """Train a Linear Regression model based on CWV

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        pandas Series: A Series with the predictions from the linear model, named GLM_CWV
    """
    logger.info("Training linear model with CWV feature")
    X = features[["CWV"]].dropna()

    X, y = check_overlapping_dates(target, X)

    model = LinearRegression()

    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "GLM_CWV"
    model = train_full_model(X, y)

    return model, predictions


def tss_cross_val_predict(X, y, model, min_train=7):
    """Apply a form of Time Series cross validation with the given data and for the given model
    We expand the data by a week in each fold and retrain the model to generate predictions for the next week.

    Args:
        X (pandas DataFrame): A DataFrame with features
        y (pandas DataFrame): A DataFrame with the target
        model (a sklearn Model): A model object with a fit and predict function
        min_train (int, optional): Number of historical values necessary to start the training cadence. Defaults to 7.

    Returns:
        pandas Series: A Series with the predictions from all the folds
    """
    test_predictions = []
    # weekly window
    nsplits = abs(round((X.index.min() - X.index.max()).days / 7))
    tscv = TimeSeriesSplit(n_splits=nsplits)

    for train_index, test_index in tscv.split(X):

        if len(train_index) < min_train:
            continue

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]

        model.fit(X_train, y_train)

        test_predictions.append(model.predict(X_test))

    test_predictions = np.array(test_predictions).flatten()
    # samples from the first training iteration don't have any test predictions so should be discarded
    num_samples_train_first_iteration = X.shape[0] - test_predictions.shape[0]
    test_predictions = pd.Series(
        test_predictions, index=X.index[num_samples_train_first_iteration:]
    )

    return test_predictions


def train_full_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    return model


def train_ldz_diff(target, features):
    """Train a model where the target is the difference in demand and the features consist of the difference in CWV

    Args:
        target (pandas DataFrame): A dataframe with a column for the actual LDZ demand
        features (pandas DataFrame): A dataframe with a column LDZ_DEMAND_DIFF and a column CWV_DIFF

    Returns:
        pandas Series: A series with predictions for the LDZ demand diff model
    """
    logger.info(
        "Training prophet model with CWV_DIFF as feature and DEMAND_DIFF as target"
    )

    # prophet and pandas don't like each other
    warnings.simplefilter("ignore", FutureWarning)

    data_input = features[["LDZ_DEMAND_DIFF", "CWV_DIFF"]].dropna()

    overlapping_dates = target.index.intersection(data_input.index)
    y = target.loc[overlapping_dates].copy()
    data_input = data_input.loc[overlapping_dates].reset_index()

    data_input.columns = ["ds", "y", "CWV_DIFF"]
    min_date = data_input["ds"].min()
    max_date = data_input["ds"].max()

    result = []
    # We've tried to align the method of model fitting to the method used for the Linear Regression
    # So this is meant to mimic a weekly retraining cycle
    # Each time the model is refitted on all historical data up to that point
    for cutoff_date in pd.date_range(
        min_date, max_date, freq="7D", inclusive="neither"
    ):
        training_data = data_input[data_input["ds"] < cutoff_date]

        model = Prophet(
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01,
            yearly_seasonality=False,
            daily_seasonality=False,
        )
        model.add_country_holidays(country_name="UK")
        model.add_regressor(
            "CWV_DIFF", mode="additive", prior_scale=0.01, standardize=False
        )

        with suppress_stdout_stderr():
            model = model.fit(training_data)

        # this is where it gets tricky
        # we're sort of pretending these are day ahead forecasts
        # previous exploration into this has shown that whether it's a daily or weekly horizon makes little difference
        horizon = 7
        if (max_date - cutoff_date).days + 1 < horizon:
            horizon = (max_date - cutoff_date).days + 1

        future_date = model.make_future_dataframe(
            periods=horizon, include_history=False
        )
        mask = data_input["ds"].between(
            cutoff_date, cutoff_date + dt.timedelta(days=horizon - 1)
        )
        future_date["CWV_DIFF"] = data_input.loc[mask, "CWV_DIFF"].values
        preds = model.predict(future_date)[["ds", "yhat"]]
        result.append(preds)

    predictions = (
        pd.concat(result)
        .rename(columns={"yhat": "PROPHET_DIFF_DEMAND", "ds": "GAS_DAY"})
        .set_index("GAS_DAY")
    )

    # putting the demand diff predictions back onto the same scale as the actual demand
    predictions["PROPHET_DIFF_DEMAND"] += y["LDZ"].shift(2)

    # train full model
    model = Prophet(
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=0.01,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    model.add_country_holidays(country_name="UK")
    model.add_regressor(
        "CWV_DIFF", mode="additive", prior_scale=0.01, standardize=False
    )

    with suppress_stdout_stderr():
        model = model.fit(data_input)

    return model, predictions["PROPHET_DIFF_DEMAND"]


def get_ldz_match_predictions(target, features):
    """Calculate the predictions according to a match heuristic.
    We match the CWV forecast to a historical CWV. However for the christmas bank holidays we take the historical average from the same period.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a columns for the CWV forecast and indicator columns for the bank holidays and work days

    Returns:
        pandas DataFrame: A DataFrame with the predictions in a column named LDZ_MATCHED
    """
    logger.info("Calculating LDZ MATCH predictions")

    data_input = pd.concat([target, features], axis=1).dropna()
    data_input["CWV_rounded"] = np.round(data_input["CWV"], decimals=1)
    data_input["MONTH"] = data_input.index.month
    data_input["DAY"] = data_input.index.day

    # this is rather arbitrary but instead of throwing away loads
    # of data we can let it give nonsense predictions for a while
    # also this allows us to compare the performance over the same period
    # as other models
    min_date = data_input.index.min() + dt.timedelta(
        days=23
    )  # 23 + 7 days from weekly retraining gives 30
    max_date = data_input.index.max()

    result = []
    # We've tried to align the method of model fitting to the method used for the Linear Regression
    # So this is meant to mimic a weekly retraining cycle
    # It may not make sense for this heuristic but it is more fair when comparing with the other methods
    for cutoff_date in pd.date_range(
        min_date, max_date, freq="7D", inclusive="neither"
    ):
        training_data = data_input[data_input.index < cutoff_date].copy()
        test_data = data_input[
            (data_input.index >= cutoff_date)
            & (data_input.index < cutoff_date + dt.timedelta(days=7))
        ].copy()

        test_data = add_average_demand_by_cwv(test_data, training_data)
        test_data = add_average_demand_by_month_day(test_data, training_data)

        mask = (
            (test_data["CHRISTMAS_DAY"])
            | (test_data["NEW_YEARS_DAY"])
            | (test_data["NEW_YEARS_EVE"])
            | (test_data["BOXING_DAY"])
        )
        test_data["LDZ_MATCHED"] = np.where(
            mask, test_data["AVERAGE_DEMAND_MONTH_DAY"], test_data["AVERAGE_DEMAND_CWV"]
        )

        # replace missing values with closest values
        for index, row in test_data.iterrows():
            if pd.isna(row["LDZ_MATCHED"]):
                # try to find the closest matching CWV_rounded value from the training set
                closest_cwv_index = (
                    (training_data["CWV_rounded"] - row["CWV_rounded"]).abs().idxmin()
                )
                test_data.at[index, "LDZ_MATCHED"] = training_data.loc[
                    closest_cwv_index, "LDZ"
                ].mean()

        result.append(test_data[["LDZ_MATCHED"]])

    predictions = pd.concat(result)
    return predictions


def add_average_demand_by_cwv(test_data, input_data):
    """Calculate the average demand by CWV and a few other variables from the given input_data and add it to the given test_data

    Args:
        test_data (pandas DataFrame): A DataFrame with columns for the CWV forecast and indicator columns for the bank holidays and workdays
        input_data (pandas DataFrame): A DataFrame with columns for the CWV forecast and indicator columns for the bank holidays and workdays

    Returns:
        pandas DataFrame: The given test_data but with an additional AVERAGE_DEMAND_CWV column
    """
    aggregated_value = (
        input_data.groupby(
            [
                "CWV_rounded",
                "WORK_DAY",
                "CHRISTMAS_DAY",
                "NEW_YEARS_DAY",
                "NEW_YEARS_EVE",
                "BOXING_DAY",
            ]
        )["LDZ"]
        .mean()
        .reset_index()
        .rename(columns={"LDZ": "AVERAGE_DEMAND_CWV"})
    )
    result = test_data.merge(aggregated_value, how="left")
    result = result.set_index(test_data.index)
    return result


def add_average_demand_by_month_day(test_data, input_data):
    """Calculate the average demand by month and day from the given input_data and add it to the given test_data

    Args:
        test_data (pandas DataFrame): A DataFrame with columns for month and day
        input_data (pandas DataFrame): A DataFrame with columns for month and day

    Returns:
        pandas DataFrame: The given test_data but with an additional AVERAGE_DEMAND_MONTH_DAY column
    """
    aggregated_value = (
        input_data.groupby(["MONTH", "DAY"])["LDZ"]
        .mean()
        .reset_index()
        .rename(columns={"LDZ": "AVERAGE_DEMAND_MONTH_DAY"})
    )
    result = test_data.merge(aggregated_value, how="left")
    result = result.set_index(test_data.index)
    return result


def train_ldz_stack_model(target, demand_diff_predictions, match_predictions):
    """Train a stacked TheilSenRegression model based on the predictions from the LDZ Match and Prophet models

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        demand_diff_predictions (pandas DataFrame): A DataFrame with predictions from the Prophet model in a column named PROPHET_DIFF_DEMAND
        match_predictions (pandas DataFrame): A DataFrame with predictions from the LDZ Match model in a column named LDZ_MATCHED

    Returns:
        pandas Series: A Series with the predictions, named LDZ_STACK
    """
    logger.info(
        "Training stacked TheilSenRegression model with LDZ MATCH and PROPHET_DIFF_DEMAND as features"
    )
    features = pd.concat([demand_diff_predictions, match_predictions], axis=1)

    features["frcst_diff_2"] = (
        features["PROPHET_DIFF_DEMAND"] - features["LDZ_MATCHED"]
    ).pow(2)
    features = features.dropna()

    X, y = check_overlapping_dates(target, features)

    model = TheilSenRegressor(fit_intercept=False, random_state=20230131)

    predictions = tss_cross_val_predict(X, y["LDZ"], model)
    predictions.name = "LDZ_STACK"

    model = TheilSenRegressor(fit_intercept=False, random_state=20230131).fit(X, y["LDZ"])

    return model, predictions


def check_overlapping_dates(dataset_one, dataset_two):
    """Determine the overlapping dates from the given datasets and filter them both by it

    Args:
        dataset_one (pandas DataFrame): A DataFrame with dates on the index
        dataset_two (pandas DataFrame): A DataFrame with dates on the index

    Returns:
        tuple: A tuple of DataFrame, in reverse order from the input (just to confuse you)
    """
    overlapping_dates = dataset_one.index.intersection(dataset_two.index)
    d1 = dataset_one.loc[overlapping_dates].copy()
    d2 = dataset_two.loc[overlapping_dates].copy()

    return d2, d1
