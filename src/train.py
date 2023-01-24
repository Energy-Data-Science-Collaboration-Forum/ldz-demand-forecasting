import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from prophet import Prophet


def train(target, features):

    overlapping_dates = target.index.intersection(features.index)
    y = target.loc[overlapping_dates].copy()
    X = features.loc[overlapping_dates].copy()

    predictions = tss_cross_val_predict(X, y)
    model = train_full_model(X, y)

    return model, predictions


def tss_cross_val_predict(X, y, min_train=7):

    test_predictions = []
    # weekly window
    nsplits = abs(round((X.index.min() - X.index.max()).days / 7))
    tscv = TimeSeriesSplit(n_splits=nsplits)
    model = LinearRegression()

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
    test_predictions = pd.Series(test_predictions, index=X.index[num_samples_train_first_iteration:])

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

    overlapping_dates = target.index.intersection(features.index)
    y = target.loc[overlapping_dates].copy()
    X = features.loc[overlapping_dates].copy()

    data_input = X[["LDZ_DEMAND_DIFF", "CWV_DIFF"]].reset_index().dropna()
    data_input.columns = ["ds", "y", "CWV_DIFF"]

    min_date = data_input["ds"].min()
    max_date = data_input["ds"].max()

    result = []
    # We've tried to align the method of model fitting to the method used for the Linear Regression
    # So this is meant to mimic a weekly retraining cycle
    # Each time the model is refitted on all historical data up to that point
    for cutoff_date in pd.date_range(min_date, max_date, freq="7D", inclusive="neither"):
        training_data = data_input[data_input['ds'] < cutoff_date]
        
        model = Prophet(
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01,
            yearly_seasonality=False,
            daily_seasonality=False,
        )
        model.add_country_holidays(country_name="UK")
        model.add_regressor("CWV_DIFF", mode="additive", prior_scale=0.01, standardize=False)
        
        model = model.fit(training_data)

        # this is where it gets tricky
        # we're sort of pretending these are day ahead forecasts
        # previous exploration into this has shown that whether it's a daily or weekly horizon makes little difference 
        horizon = 7
        if (max_date - cutoff_date).days + 1 < horizon:
            horizon = (max_date - cutoff_date).days + 1

        future_date = model.make_future_dataframe(periods=horizon, include_history=False)
        mask = data_input['ds'].between(cutoff_date, cutoff_date + dt.timedelta(days=horizon-1))
        future_date['CWV_DIFF'] = data_input.loc[mask, 'CWV_DIFF'].values
        preds = model.predict(future_date)[['ds', 'yhat']]
        result.append(preds)
    
    predictions = (pd.concat(result).
                    rename(columns={'yhat':'PREDICTION', 'ds':'GAS_DAY'}).
                    set_index('GAS_DAY'))

    # putting the demand diff predictions back onto the same scale as the actual demand
    predictions['PREDICTION'] += y["LDZ"].shift(2)

    # train full model
    model = Prophet(
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01,
            yearly_seasonality=False,
            daily_seasonality=False,
    )
    model.add_country_holidays(country_name="UK")
    model.add_regressor("CWV_DIFF", mode="additive", prior_scale=0.01, standardize=False)
    
    model = model.fit(data_input)
    
    return model, predictions['PREDICTION']

