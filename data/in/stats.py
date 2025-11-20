import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = "cattle_data_train.csv"   # <- SET THIS
TARGET = "Milk_Yield_L"        # <- SET YOUR LABEL HERE


# ============================================================
# 1) LOAD RAW
# ============================================================
df = pd.read_csv(TRAIN_PATH)
print("\nRAW shape:", df.shape)

y = df[TARGET]
X = df.drop(columns=[TARGET])

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

X = X.drop (DROP_FEATURES, axis = 1)
def month_to_season (m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

months = pd.to_datetime (X['Date']).dt.month
X = X.drop (columns = ['Date'])
X['Date'] = months.apply (month_to_season)


# ============================================================
# 2) ENCODE NON-NUMERIC
# ============================================================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

if len(categorical_cols) > 0:
    print("\nEncoding categorical:", list(categorical_cols))
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("\nNo object-dtype columns found.")


# ============================================================
# 3) IMPUTE ALL MISSING VALUES
# ============================================================
print("\nMissing value counts BEFORE imputation:")
print(df.isna().sum()[df.isna().sum() > 0])

num_cols = X.select_dtypes(include=[np.number]).columns

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])

print("\nMissing value counts AFTER imputation:")
print(pd.DataFrame(X).isna().sum().sum(), "total missing (should be 0)")


# ============================================================
# 4) TRAIN/TEST SPLIT + SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =========
# 4) TEST
# =========

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Train model
gbr = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=None,
    max_leaf_nodes=32,
    min_samples_leaf=200,
    max_iter=800,
    l2_regularization=1.0,
    early_stopping=False
)
gbr.fit(X_train, y_train)

# Predict
y_pred = gbr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("Train RMSE:", rmse)

y_pred = gbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)

print("True Mean:", y_train.mean())
print("Pred Mean:", y_pred.mean())