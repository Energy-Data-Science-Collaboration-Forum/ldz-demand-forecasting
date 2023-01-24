import joblib
from datetime import datetime as dt
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_gas_features,
)
from src.train import train
from src.evaluate import evaluate_models

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {"CWV": "data/cwv_data_20221118_214135.csv", 
            "GAS_DEMAND": "data/gas_actuals_20221118_214136.csv"}
ACTUALS = {"GAS": "data/gas_actuals_20221118_214136.csv"}

gas_demand_actuals = prepare_gas_demand_actuals(ACTUALS["GAS"])

gas_features = prepare_gas_features(FEATURES)

ldz_model, ldz_cv_predictions = train(gas_demand_actuals[["LDZ"]], gas_features)
joblib.dump(ldz_model, f"data/ldz_model_{dt.now().strftime(format=FORMAT)}.joblib")


model_performance = evaluate_models(
    ldz_cv_predictions, gas_demand_actuals
)
model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

print(model_performance)
