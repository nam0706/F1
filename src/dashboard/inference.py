from __future__ import annotations

import pandas as pd


def probability_frame(model_package: dict, row: pd.DataFrame) -> pd.DataFrame:
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    class_labels = {int(key): value for key, value in model_package["class_labels"].items()}
    probabilities = model.predict_proba(row[feature_columns])[0]
    classes = [int(value) for value in model.named_steps["classifier"].classes_]
    return pd.DataFrame(
        {
            "class_id": classes,
            "finish_bucket": [class_labels[class_id] for class_id in classes],
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)
