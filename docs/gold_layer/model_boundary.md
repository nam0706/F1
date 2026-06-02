# Gold vs Model Boundary

## Gold Responsibilities

Gold owns domain and temporal feature engineering:

```text
labels
lap-level feature grain
shifted previous-lap features
rolling pace features
pit state
stint/tyre state
interval/gap state
position state
weather context
overtake context
horizon datasets
leakage audit
feature contract
```

Gold may write multiple training-ready datasets, but it must not fit model-specific transformers.

## Model Training Responsibilities

Model training owns:

```text
target selection
horizon selection
train/validation/test split
feature column selection from contract
imputation fit on train only
scaling fit on train only
encoding fit on train only
class balancing
model training
hyperparameter search
evaluation
artifact serialization
```

## Prohibited in Model Training

Model training must not create F1 domain features such as:

```text
rolling_avg_lap_5
pit_count_so_far
current_tyre_age
laps_since_last_pit
drs_train_flag
race_progress_pct
```

If a feature has domain meaning or time semantics, it belongs in Gold.

## Model Input Contract

Training code must load:

```text
data/gold/training/*.parquet
data/gold/metadata/feature_contract.json
data/gold/metadata/label_contract.json
data/gold/metadata/leakage_report.json
```

The model step may only use columns where:

```text
model_allowed = true
```
