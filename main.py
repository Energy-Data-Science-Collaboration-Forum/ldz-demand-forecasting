import joblib
from datetime import datetime as dt
import pandas as pd
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_gas_features,
)
from src.train import train, train_ldz_diff, get_ldz_match_predictions
from src.evaluate import evaluate_models

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {
    "CWV": "data/cwv_data_20221118_214135.csv",
    "GAS_DEMAND": "data/gas_actuals_20221118_214136.csv",
}

ldz_demand_actuals = prepare_gas_demand_actuals("data/gas_actuals_20221118_214136.csv")
ldz_demand_actuals = ldz_demand_actuals[["LDZ"]]

gas_features = prepare_gas_features(FEATURES)

ldz_model, ldz_cv_predictions = train(ldz_demand_actuals, gas_features)
joblib.dump(ldz_model, f"data/ldz_model_{dt.now().strftime(format=FORMAT)}.joblib")

ldz_diff_model, ldz_diff_cv_predictions = train_ldz_diff(
    ldz_demand_actuals, gas_features
)
joblib.dump(
    ldz_diff_model, f"data/ldz_diff_model_{dt.now().strftime(format=FORMAT)}.joblib"
)

ldz_match_predictions = get_ldz_match_predictions(ldz_demand_actuals, gas_features)

all_predictions = pd.concat(
    [ldz_cv_predictions, ldz_diff_cv_predictions, ldz_match_predictions], axis=1
)

model_performance = evaluate_models(all_predictions, ldz_demand_actuals)
model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

print(model_performance)
