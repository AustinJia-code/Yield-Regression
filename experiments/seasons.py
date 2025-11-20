import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings ('ignore', category = ConvergenceWarning)

# ------------------------
# Load data
# ------------------------
train_data = pd.read_csv("../data/in/cattle_data_train.csv")

# Ensure Date is datetime
train_data["Date"] = pd.to_datetime(train_data["Date"], errors="coerce")
if train_data["Date"].isna().any():
    print("Warning: some dates could not be parsed and will be dropped.")
    train_data = train_data.dropna(subset=["Date"])

# Day of year
train_data["DayOfYear"] = train_data["Date"].dt.dayofyear

# ------------------------
# Map month to season
# ------------------------
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

train_data["Season"] = train_data["Date"].dt.month.apply(month_to_season)

# ------------------------
# Shift Winter days (Dec) to front
# ------------------------
train_data["DayOfYear_shifted"] = train_data["DayOfYear"]
mask_winter = train_data["Season"] == "Winter"
train_data.loc[mask_winter & (train_data["DayOfYear"] >= 335), "DayOfYear_shifted"] -= 365

# ------------------------
# Define features
# ------------------------
numeric_features = ["DayOfYear_shifted", "Age_Months",
    "Weight_kg",
    "Parity",
    "Days_in_Milk",
    "Feed_Quantity_kg",
    "Water_Intake_L",
    "Ambient_Temperature_C",
    "Previous_Week_Avg_Yield",]
categorical_features = [ "Lactation_Stage",
    "Milking_Interval_hrs",]
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
    "Date",
    "Season",
    "DayOfYear"
]

TARGET = "Milk_Yield_L"

# ------------------------
# Split train/test
# ------------------------
train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)

# ------------------------
# Train per-season neural nets
# ------------------------
season_models = {}
season_rmses = {}
all_y_test = []
all_y_pred = []

for season in train_data["Season"].unique(): 
    train_season = train_df[train_df["Season"] == season]
    test_season = test_df[test_df["Season"] == season]
 
    if len(train_season) == 0 or len(test_season) == 0: 
        continue 
 
    # Encode categorical features 
    X_train = pd.get_dummies(train_season, columns = categorical_features, drop_first=True).drop (DROP_FEATURES, axis = 1)
    X_train = X_train.drop ("Milk_Yield_L", axis = 1)
    # if (season != "Winter"):
    #     X_train.drop ("DayOfYear_shifted")
    print (X_train.columns)
    X_test = pd.get_dummies(test_season, columns = categorical_features, drop_first=True).drop (DROP_FEATURES, axis = 1)
 
    median_val = X_train["Feed_Quantity_kg"].median() 
    X_train.loc[X_train["Feed_Quantity_kg"].isna(), "Feed_Quantity_kg"] = median_val 
    X_test.loc[X_test["Feed_Quantity_kg"].isna(), "Feed_Quantity_kg"] = median_val 
 
    # Align columns 
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0) 
 
    # Scale numeric features 
    scaler = StandardScaler() 
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features]) 
    X_test[numeric_features] = scaler.transform(X_test[numeric_features]) 
 
    y_train = train_season[TARGET].values 
    y_test = test_season[TARGET].values 
 
    # Train neural network with warm start iterations
    print(f"\n{'='*60}")
    print(f"Training Season: {season}")
    print(f"{'='*60}")
    
    model = MLPRegressor(hidden_layer_sizes=(64, 64, 64), 
                        activation="tanh", 
                        learning_rate_init=0.0003, 
                        learning_rate="adaptive", 
                        early_stopping=False,  
                        n_iter_no_change=20, 
                        verbose=False, 
                        warm_start=True, 
                        max_iter=10,  # Train 10 iterations at a time
                        random_state=1)

    prev_rmse = float ('inf')
    test_rmse = 0
    TOLERANCE = 0.00005
    print (model)
    iterations_per_step = 10
    total_iterations = 0
    # total_iterations < 200
    while (total_iterations < 200):
        if test_rmse > 0:
            prev_rmse = test_rmse
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_loss = root_mean_squared_error(y_train, train_pred)
        test_loss = root_mean_squared_error(y_test, test_pred)
        
        total_iterations += iterations_per_step
        current_iter = total_iterations
        print(f"Iter {current_iter:3d}/{total_iterations} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Test MSE: {test_loss:.4f} | "
              f"Loss: {model.loss_:.6f}")
    
    y_pred = model.predict(X_test)
    season_models[season] = (model, scaler, X_train.columns)
    print(f"{'='*60}\n")

    all_y_test.append(pd.Series(y_test, index=test_season.index))
    all_y_pred.append(pd.Series(y_pred, index=test_season.index))

# ------------------------
# Global RMSE
# ------------------------
all_y_test = pd.concat(all_y_test)
all_y_pred = pd.concat(all_y_pred)
global_rmse = root_mean_squared_error(all_y_test, all_y_pred)
print(f"Global RMSE (all seasons): {global_rmse:.3f}")

# ------------------------
# Plot daily averages + predictions
# ------------------------
daily_avg = train_data.groupby(["DayOfYear_shifted", "Season"])[TARGET].mean().reset_index()

# Original colors for averages
season_colors = {
    "Winter": "blue",
    "Spring": "green",
    "Summer": "orange",
    "Fall": "brown"
}

# Darker colors for predictions
pred_colors = {
    "Winter": "navy",
    "Spring": "darkgreen",
    "Summer": "darkorange",
    "Fall": "saddlebrown"
}

plt.figure(figsize=(12,5))

# Plot actual averages
for season, color in season_colors.items():
    season_data = daily_avg[daily_avg["Season"] == season]
    plt.plot(season_data["DayOfYear_shifted"], season_data[TARGET], 
             marker="o", color=color, label=f"{season} (avg)")

# Overlay predictions
for season, (model, scaler, columns) in season_models.items():
    season_days = np.array(daily_avg[daily_avg["Season"] == season]["DayOfYear_shifted"]).reshape(-1,1)
    # Build full feature dataframe for prediction
    X_pred = pd.DataFrame(season_days, columns=["DayOfYear_shifted"])
    X_pred["Feed_Quantity_kg"] = train_data["Feed_Quantity_kg"].median()
    for col in columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[columns]

    X_pred_scaled = scaler.transform(X_pred)
    y_pred = model.predict(X_pred_scaled)
    plt.plot(season_days, y_pred, linestyle="--", color=pred_colors[season], label=f"{season} (pred)")

plt.xlabel("Day of Year (Winter shifted)")
plt.ylabel("Milk Yield (L)")
plt.title("Milk Yield by Day of Year with Per-Season Neural Net")
plt.legend()
plt.grid(True)
plt.show()
