from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

###### PARAMS ######
# Data Cleaning
LABEL_COL = 'Milk_Yield_L'
ALL_FEATURES = ["Cattle_ID", "Breed", "Climate_Zone", "Management_System", 
                "Age_Months", "Weight_kg", "Parity", "Lactation_Stage", 
                "Days_in_Milk", "Feed_Type", "Feed_Quantity_kg",
                "Feeding_Frequency", "Water_Intake_L", "Walking_Distance_km", 
                "Grazing_Duration_hrs", "Rumination_Time_hrs", "Resting_Hours",
                "Ambient_Temperature_C", "Humidity_percent", "Housing_Score",
                "FMD_Vaccine", "Brucellosis_Vaccine", "HS_Vaccine", 
                "BQ_Vaccine", "Anthrax_Vaccine", "IBR_Vaccine", "BVD_Vaccine",
                "Rabies_Vaccine", "Previous_Week_Avg_Yield", 
                "Body_Condition_Score", "Milking_Interval_hrs", "Date",
                "Farm_ID", "Feed_Quantity_lb", "Mastitis", "Milk_Yield_L"]

DROP_FEATURES = ['Cattle_ID',
                 'Feed_Quantity_kg',
                 'Farm_ID',                # Too many options for one-hot
                 ]

IMPUTE_FEATURES = ['Feed_Quantity_lb',     # Missing 10k
                 'Housing_Score'           # Missing 6k
                 ]

# Feature Engineering
CATEGORICAL_FEATURES = ['Breed', 'Climate_Zone', 'Management_System',
                        'Feed_Type', 'Lactation_Stage']


######## FUNCTIONS ########
def clean_data (
    raw_data: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Drop IDs, redundant features, and labels
    Returns tuple (labels, cleaned_data)
    """
    labels = None

    # Drop
    cleaned_data = raw_data.drop (DROP_FEATURES,
                                  axis = 1)
    if LABEL_COL in cleaned_data:
        labels = raw_data[LABEL_COL].values.ravel ()
        cleaned_data = cleaned_data.drop (LABEL_COL, axis = 1)
    
    '''
    # KNN Impute with numeric features
    numeric_features = cleaned_data.select_dtypes (include = [np.number]).columns
    scaler = StandardScaler ()
    scaled = scaler.fit_transform (cleaned_data[numeric_features])

    imputer = KNNImputer (n_neighbors = 5)
    imputed_scaled = imputer.fit_transform (scaled)
    cleaned_data[numeric_features] = imputed_scaled

    raw_data[IMPUTE_FEATURES] = imputer_knn.fit_transform (raw_data[IMPUTE_FEATURES])
    '''
    
    # Mean impute
    imputer_mean = SimpleImputer (missing_values = np.nan, strategy = 'median') 
    for feature in IMPUTE_FEATURES:
        raw_data[feature] = imputer_mean.fit_transform (raw_data[[feature]])

    # Return
    return labels, cleaned_data


def engineer_data (
    cleaned_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Feature Engineering
    Returns engineered dataframe
    """
    # Convert date to monthly circular representation
    months = pd.to_datetime (cleaned_data['Date']).dt.month
    engineered_data = cleaned_data.drop (columns = ['Date'])
    engineered_data['Month_sin'] = np.sin (2 * np.pi * months / 12)
    engineered_data['Month_cos'] = np.cos (2 * np.pi * months / 12)

    # One Hot encode
    engineered_data = pd.get_dummies (engineered_data,
                                      columns = CATEGORICAL_FEATURES,
                                      drop_first = True)

    return engineered_data


if __name__ == '__main__':
    pass