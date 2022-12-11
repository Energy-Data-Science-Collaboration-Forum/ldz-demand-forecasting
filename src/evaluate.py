import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def evaluate_models(ldz_predictions, actuals):
    ldz_predictions.name = "LDZ_PREDICTIONS"
    combined = pd.concat([ldz_predictions, actuals], axis=1)

    combined = combined.dropna()

    result = pd.DataFrame(
        {
            "MODEL": ["LDZ"],
            "MAE": [
                mean_absolute_error(combined["LDZ"], combined["LDZ_PREDICTIONS"]),
            ],
            "MAPE": [
                mean_absolute_percentage_error(
                    combined["LDZ"], combined["LDZ_PREDICTIONS"]
                ),
            ],
        }
    )

    return result
