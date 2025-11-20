import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

import os
if 'experiments' in os.getcwd ():
    os.chdir (os.getcwd () + "/..")
import pandas as pd

DROP_FEATURES = [
    "Cattle_ID",
    "Farm_ID",
    "Feed_Quantity_lb",
    "Breed",
    "Climate_Zone",
    "Management_System",
    "Feed_Type",
    "Feeding_Frequency",
    "Walking_Distance_km",
    "Grazing_Duration_hrs",
    "Rumination_Time_hrs",
    "Resting_Hours",
    "Body_Condition_Score",
    "Humidity_percent",
    "BVD_Vaccine",
    "FMD_Vaccine",
    "Brucellosis_Vaccine",
    "HS_Vaccine",
    "BQ_Vaccine",
    "Housing_Score",
]

CATEGORICAL_FEATURES = [
    "Lactation_Stage",
    "Milking_Interval_hrs",
    # "IBR_Vaccine",
    # "Anthrax_Vaccine",
    # "Rabies_Vaccine"
]

train_data = pd.read_csv ("./data/in/cattle_data_train.csv")

# Example: df is your original dataframe
df = train_data.copy()  # replace with your dataframe

# Ensure Date is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Map month to season
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["Season"] = df["Date"].dt.month.apply(month_to_season)

# Drop Date column (we can encode season as a categorical variable later if needed)
df = df.drop(columns=["Date"])
df = df.drop (DROP_FEATURES, axis = 1)
df = pd.get_dummies (df, columns = CATEGORICAL_FEATURES, 
                             drop_first = True)


# Separate features and target
TARGET = "Milk_Yield_L"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode season as one-hot for regressors if needed (or keep as string to split)
# Here we'll split data by season
seasons = df["Season"].unique()

# Split into train/test globally
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

median_val = X_train_full["Feed_Quantity_kg"].median ()
X_train_full.loc[X_train_full["Feed_Quantity_kg"].isna (), "Feed_Quantity_kg"] = median_val
median_val = X_test_full["Feed_Quantity_kg"].median ()
X_test_full.loc[X_test_full["Feed_Quantity_kg"].isna (), "Feed_Quantity_kg"] = median_val


# Prepare a dict to hold models and RMSEs
season_models = {}
season_rmses = {}

# Train a separate regressor per season
for season in seasons:
    # Mask for current season
    train_mask = X_train_full["Season"] == season
    test_mask = X_test_full["Season"] == season

    X_train = X_train_full[train_mask].drop(columns=["Season"])
    y_train = y_train_full[train_mask]

    X_test = X_test_full[test_mask].drop(columns=["Season"])
    y_test = y_test_full[test_mask]

    # Initialize a regressor (MLP here, can be replaced)
    model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         max_iter=500,
                         random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    season_models[season] = model
    season_rmses[season] = rmse
    print(f"{season} RMSE: {rmse:.3f}")

# Optional: global weighted RMSE
all_y_test = []
all_y_pred = []

for season in seasons:
    test_mask = X_test_full["Season"] == season
    X_test = X_test_full[test_mask].drop(columns=["Season"])
    y_test = y_test_full[test_mask]
    if len(X_test) == 0:
        continue
    y_pred = season_models[season].predict(X_test)
    all_y_test.append(y_test)
    all_y_pred.append(pd.Series(y_pred, index=y_test.index))

all_y_test = pd.concat(all_y_test)
all_y_pred = pd.concat(all_y_pred)
global_rmse = root_mean_squared_error(all_y_test, all_y_pred)
print(f"Global RMSE (all seasons): {global_rmse:.3f}")

# for now, change this to just predict the mean of the season