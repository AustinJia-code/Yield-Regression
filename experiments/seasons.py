import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Load data
# ------------------------
train_data = pd.read_csv("../data/in/cattle_data_train.csv")

train_data["Date"] = pd.to_datetime(train_data["Date"], errors="coerce")
train_data = train_data.dropna(subset=["Date"])

train_data["DayOfYear"] = train_data["Date"].dt.dayofyear

# ------------------------
# CLEAN DATA: Remove negative and extreme yields
# ------------------------
print("="*60)
print("DATA CLEANING")
print("="*60)
print(f"Original data size: {len(train_data)}")
print(f"Samples with negative yield: {(train_data['Milk_Yield_L'] < 0).sum()}")
print(f"Samples with extreme yield (>50L): {(train_data['Milk_Yield_L'] > 50).sum()}")

train_data = train_data[train_data['Milk_Yield_L'] > 0]  # Remove negative
train_data = train_data[train_data['Milk_Yield_L'] < 50]  # Remove extremes

print(f"Cleaned data size: {len(train_data)}")
print(f"Yield range after cleaning: {train_data['Milk_Yield_L'].min():.2f} - {train_data['Milk_Yield_L'].max():.2f}")
print("="*60 + "\n")

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

# =======================================================
# PLOT: Feed Quantity vs Milk Yield (Spring only)
# =======================================================
spring_data = train_data[train_data["Season"] == "Spring"].copy()

# Separate missing vs present feed data
missing_feed = spring_data["Feed_Quantity_kg"].isna()
spring_data["Feed_Quantity_kg_plot"] = spring_data["Feed_Quantity_kg"].fillna(0)

has_feed = spring_data[~missing_feed]
no_feed = spring_data[missing_feed]

plt.figure(figsize=(10, 6))

# Plot samples WITH feed data
plt.scatter(
    has_feed["Feed_Quantity_kg_plot"], 
    has_feed["Milk_Yield_L"],
    alpha=0.4, 
    edgecolor="k",
    linewidth=0.5,
    label=f"Has Feed Data (n={len(has_feed)})",
    color="steelblue"
)

# Plot samples WITHOUT feed data (at x=0)
plt.scatter(
    no_feed["Feed_Quantity_kg_plot"], 
    no_feed["Milk_Yield_L"],
    alpha=0.4, 
    edgecolor="k",
    linewidth=0.5,
    label=f"Missing Feed Data (n={len(no_feed)})",
    color="orange",
    marker="^"
)

plt.xlabel("Feed Quantity (kg)", fontsize=12)
plt.ylabel("Milk Yield (L)", fontsize=12)
plt.title("Feed Quantity vs Milk Yield - Spring Season", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nSpring Season Feed Data Summary:")
print(f"Total samples: {len(spring_data)}")
print(f"Missing feed data: {missing_feed.sum()} ({100*missing_feed.sum()/len(spring_data):.1f}%)")
print(f"Has feed data: {(~missing_feed).sum()} ({100*(~missing_feed).sum()/len(spring_data):.1f}%)")
print(f"\nFeed Quantity Stats (non-missing):")
print(has_feed["Feed_Quantity_kg"].describe())

# ------------------------
# Safe-Drops-Only Feature Setup
# ------------------------
numeric_features = [
    "Age_Months",
    "Weight_kg",
    "Days_in_Milk",
    "Feed_Quantity_kg",
    "Water_Intake_L",
    "Ambient_Temperature_C",
    "Previous_Week_Avg_Yield",
]

categorical_features = [
    "Lactation_Stage",
    "Milking_Interval_hrs",
    "Breed",
    "Parity",
]

DROP_FEATURES = [
    "DayOfYear_shifted",
    "Cattle_ID",
    "Farm_ID",
    "Feed_Quantity_lb",
    "Climate_Zone",
    "Management_System",
    "Feed_Type",
    "Feeding_Frequency",
    "Resting_Hours",
    "Housing_Score",
    "DayOfYear",
    "Date",
    "Season",
    "Rumination_Time_hrs",
    "Humidity_percent",
    "Grazing_Duration_hrs",
    "Walking_Distance_km",
    "Body_Condition_Score",
    "BVD_Vaccine",
    "FMD_Vaccine",
    "Brucellosis_Vaccine",
    "HS_Vaccine",
    "BQ_Vaccine",
]

TARGET = "Milk_Yield_L"

# ------------------------
# Split train/test
# ------------------------
train_df, test_df = train_test_split(train_data, test_size=0.2, random_state=42)

# ------------------------
# Yield Distribution Analysis
# ------------------------
spring_train = train_df[train_df["Season"] == "Spring"]
print("\n" + "="*60)
print("YIELD DISTRIBUTION IN TRAINING DATA (Spring)")
print("="*60)
print(spring_train[TARGET].describe())
print(f"\nSamples above 35L: {(spring_train[TARGET] > 35).sum()}")
print(f"Samples above 30L: {(spring_train[TARGET] > 30).sum()}")
print(f"Samples above 27L: {(spring_train[TARGET] > 27).sum()}")
print(f"Samples above 25L: {(spring_train[TARGET] > 25).sum()}")

# Analyze high-yield vs normal samples
high_yield = spring_train[spring_train[TARGET] > 30]
normal_yield = spring_train[spring_train[TARGET] <= 30]

print("\n" + "="*60)
print("HIGH YIELD (>30L) vs NORMAL YIELD COMPARISON")
print("="*60)
print(f"High yield samples: {len(high_yield)} ({100*len(high_yield)/len(spring_train):.2f}%)")
print(f"Normal yield samples: {len(normal_yield)} ({100*len(normal_yield)/len(spring_train):.2f}%)")
print("\nFeature Comparison:")
for feat in ["Weight_kg", "Age_Months", "Feed_Quantity_kg", "Water_Intake_L", "Previous_Week_Avg_Yield"]:
    print(f"{feat:25s}: Normal={normal_yield[feat].mean():6.2f}, High={high_yield[feat].mean():6.2f}, Diff={high_yield[feat].mean()-normal_yield[feat].mean():+6.2f}")
print("="*60)

# ------------------------
# Train Spring-only model
# ------------------------
season_models = {}
all_y_test = []
all_y_pred = []

for season in train_data["Season"].unique():
    if season != "Spring":
        continue

    train_season = train_df[train_df["Season"] == season]
    test_season = test_df[test_df["Season"] == season]

    if len(train_season) == 0 or len(test_season) == 0:
        continue

    # ========================================================
    # PREDICTIVE IMPUTATION FOR MISSING FEED
    # ========================================================
    
    # Identify which rows have missing feed
    train_feed_missing = train_season["Feed_Quantity_kg"].isna()
    test_feed_missing = test_season["Feed_Quantity_kg"].isna()
    
    print(f"\nTrain samples with missing feed: {train_feed_missing.sum()} ({100*train_feed_missing.mean():.1f}%)")
    print(f"Test samples with missing feed: {test_feed_missing.sum()} ({100*test_feed_missing.mean():.1f}%)")
    
    if train_feed_missing.sum() > 0:
        # Features to predict feed (exclude feed itself and target)
        feed_predictors = ["Weight_kg", "Age_Months", "Days_in_Milk", 
                          "Water_Intake_L", "Previous_Week_Avg_Yield",
                          "Ambient_Temperature_C", "Lactation_Stage", "Parity", "Breed"]
        
        # Get samples WITH feed data for training the imputer
        has_feed_train = train_season[~train_feed_missing].copy()
        missing_feed_train = train_season[train_feed_missing].copy()
        
        # Prepare data for feed imputation model
        X_feed_train = pd.get_dummies(has_feed_train[feed_predictors], drop_first=True)
        y_feed_train = has_feed_train["Feed_Quantity_kg"].values
        
        # Train feed imputation model
        feed_imputer = GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1,
            random_state=42
        )
        feed_imputer.fit(X_feed_train, y_feed_train)
        
        # Predict missing feed in TRAINING set
        X_feed_predict_train = pd.get_dummies(missing_feed_train[feed_predictors], drop_first=True)
        X_feed_predict_train = X_feed_predict_train.reindex(columns=X_feed_train.columns, fill_value=0)
        predicted_feed_train = feed_imputer.predict(X_feed_predict_train)
        
        print(f"Predicted train feed range: {predicted_feed_train.min():.2f} - {predicted_feed_train.max():.2f} kg")
        
        # Update train_season with predicted feed
        train_season = train_season.copy()
        train_season.loc[train_feed_missing, "Feed_Quantity_kg"] = predicted_feed_train
        
        # Predict missing feed in TEST set
        if test_feed_missing.sum() > 0:
            missing_feed_test = test_season[test_feed_missing].copy()
            X_feed_predict_test = pd.get_dummies(missing_feed_test[feed_predictors], drop_first=True)
            X_feed_predict_test = X_feed_predict_test.reindex(columns=X_feed_train.columns, fill_value=0)
            predicted_feed_test = feed_imputer.predict(X_feed_predict_test)
            
            print(f"Predicted test feed range: {predicted_feed_test.min():.2f} - {predicted_feed_test.max():.2f} kg")
            
            test_season = test_season.copy()
            test_season.loc[test_feed_missing, "Feed_Quantity_kg"] = predicted_feed_test
    
    # Now proceed with one-hot encoding
    X_train = (
        pd.get_dummies(train_season, columns=categorical_features, drop_first=True)
        .drop(DROP_FEATURES + [TARGET], axis=1)
    )
    X_test = (
        pd.get_dummies(test_season, columns=categorical_features, drop_first=True)
        .drop(DROP_FEATURES + [TARGET], axis=1)
    )
    
    # No need to impute anymore - already done above
    # But handle any remaining NaN just in case
    if X_train["Feed_Quantity_kg"].isna().sum() > 0:
        median_val = X_train["Feed_Quantity_kg"].median()
        X_train["Feed_Quantity_kg"].fillna(median_val, inplace=True)
        X_test["Feed_Quantity_kg"].fillna(median_val, inplace=True)

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    y_train = train_season[TARGET].values
    y_test = test_season[TARGET].values

    # Train Gradient Boosting Regressor
    print(f"\nTraining {season} model with Gradient Boosting...")
    
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = root_mean_squared_error(y_train, train_pred)
    test_rmse = root_mean_squared_error(y_test, test_pred)

    print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Store model
    season_models[season] = (model, scaler, X_train.columns)

    all_y_test.append(pd.Series(y_test, index=test_season.index))
    all_y_pred.append(pd.Series(test_pred, index=test_season.index))

# ------------------------
# Global RMSE
# ------------------------
all_y_test = pd.concat(all_y_test)
all_y_pred = pd.concat(all_y_pred)
global_rmse = root_mean_squared_error(all_y_test, all_y_pred)
print(f"\nGlobal RMSE (Spring only): {global_rmse:.3f}")

# Print prediction range
print(f"Actual yield range: {all_y_test.min():.2f} - {all_y_test.max():.2f}")
print(f"Predicted yield range: {all_y_pred.min():.2f} - {all_y_pred.max():.2f}")

# =======================================================
# Plot Predictions vs Actual for Spring Model
# =======================================================
spring_model, spring_scaler, spring_cols = season_models["Spring"]

# Get predictions on full Spring test set
spring_test = test_df[test_df["Season"] == "Spring"].copy()

# APPLY SAME PREDICTIVE IMPUTATION TO PLOT DATA
spring_test_feed_missing = spring_test["Feed_Quantity_kg"].isna()

if spring_test_feed_missing.sum() > 0:
    feed_predictors = ["Weight_kg", "Age_Months", "Days_in_Milk", 
                      "Water_Intake_L", "Previous_Week_Avg_Yield",
                      "Ambient_Temperature_C", "Lactation_Stage", "Parity", "Breed"]
    
    # Use samples with feed from training to build imputer
    spring_train_has_feed = train_df[(train_df["Season"] == "Spring") & train_df["Feed_Quantity_kg"].notna()]
    
    X_feed_imputer = pd.get_dummies(spring_train_has_feed[feed_predictors], drop_first=True)
    y_feed_imputer = spring_train_has_feed["Feed_Quantity_kg"].values
    
    plot_feed_imputer = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    plot_feed_imputer.fit(X_feed_imputer, y_feed_imputer)
    
    # Predict missing feed in plot data
    missing_plot_data = spring_test[spring_test_feed_missing]
    X_plot_feed_predict = pd.get_dummies(missing_plot_data[feed_predictors], drop_first=True)
    X_plot_feed_predict = X_plot_feed_predict.reindex(columns=X_feed_imputer.columns, fill_value=0)
    
    predicted_plot_feed = plot_feed_imputer.predict(X_plot_feed_predict)
    spring_test.loc[spring_test_feed_missing, "Feed_Quantity_kg"] = predicted_plot_feed

X_plot = pd.get_dummies(spring_test, columns=categorical_features, drop_first=True)
X_plot = X_plot.drop(DROP_FEATURES + [TARGET], axis=1)

# Align missing columns
X_plot = X_plot.reindex(columns=spring_cols, fill_value=0)

# Handle any remaining NaN
if X_plot["Feed_Quantity_kg"].isna().sum() > 0:
    X_plot["Feed_Quantity_kg"].fillna(
        train_df[train_df["Season"] == "Spring"]["Feed_Quantity_kg"].median(),
        inplace=True
    )

# Scale
X_plot[numeric_features] = spring_scaler.transform(X_plot[numeric_features])

y_true = spring_test[TARGET].values
y_pred = spring_model.predict(X_plot)

# ---------------------------
# Scatter Plot: Actual vs Predicted
# ---------------------------
plt.figure(figsize=(7,7))
plt.scatter(y_true, y_pred, alpha=0.25, edgecolor="k")

# Diagonal reference line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

plt.xlabel("Actual Milk Yield (L)")
plt.ylabel("Predicted Milk Yield (L)")
plt.title("Spring Model — Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Residual Plot
# ---------------------------
residuals = y_true - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.3, edgecolor="k")
plt.axhline(0, color="red", linestyle="--", linewidth=2)

plt.xlabel("Predicted Milk Yield (L)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Spring Model Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()