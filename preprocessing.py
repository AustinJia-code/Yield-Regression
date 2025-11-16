from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import zscore
from enum import Enum

class RecordType (Enum):
    KEEP = 0
    DROP = 1

###### PARAMS ######
# Data Cleaning
LABEL_COL = 'Milk_Yield_L'
NUM_FEATURES = {'Cattle_ID':                    RecordType.DROP,
                'Age_Months':                   RecordType.KEEP,    #
                'Weight_kg':                    RecordType.KEEP,    #
                'Parity':                       RecordType.KEEP,    #
                'Days_in_Milk':                 RecordType.KEEP,    #
                'Feed_Quantity_kg':             RecordType.DROP,
                'Feed_Quantity_lb':             RecordType.KEEP,    #
                'Feeding_Frequency':            RecordType.DROP,
                'Water_Intake_L':               RecordType.KEEP,    #
                'Walking_Distance_km':          RecordType.DROP, 
                'Grazing_Duration_hrs':         RecordType.DROP,
                'Rumination_Time_hrs':          RecordType.DROP,
                'Resting_Hours':                RecordType.DROP,
                'Ambient_Temperature_C':        RecordType.KEEP,    #
                'Humidity_percent':             RecordType.DROP,
                'Housing_Score':                RecordType.DROP,
                'FMD_Vaccine':                  RecordType.DROP,
                'Brucellosis_Vaccine':          RecordType.DROP, 
                'HS_Vaccine':                   RecordType.DROP, 
                'BQ_Vaccine':                   RecordType.DROP,
                'Anthrax_Vaccine':              RecordType.KEEP,    #
                'IBR_Vaccine':                  RecordType.KEEP,    #
                'BVD_Vaccine':                  RecordType.DROP,
                'Rabies_Vaccine':               RecordType.KEEP,    #
                'Previous_Week_Avg_Yield':      RecordType.KEEP,    # 
                'Body_Condition_Score':         RecordType.DROP,
                'Milking_Interval_hrs':         RecordType.KEEP,    #
                'Farm_ID':                      RecordType.DROP,
                'Mastitis':                     RecordType.KEEP,}   #

# Non-Dropped Categorical Features
CAT_FEATURES = {'Breed':                        RecordType.DROP,
                'Climate_Zone':                 RecordType.DROP,
                'Management_System':            RecordType.DROP,
                'Lactation_Stage':              RecordType.KEEP,    #
                'Feed_Type':                    RecordType.DROP,
                'Date':                         RecordType.KEEP,    #
                }


######## FUNCTIONS ########
def clean_data (
    raw_data: pd.DataFrame,
    is_train: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Drop IDs, redundant features, and labels
    Returns tuple (labels, cleaned_data)
    """
    labels = None

    # Drop useless features
    drop_fts = ([ft for ft, t in NUM_FEATURES.items () if t == RecordType.DROP] +
                [ft for ft, t in CAT_FEATURES.items () if t == RecordType.DROP])
    
    cleaned_data = raw_data.drop (drop_fts, axis = 1)
    
    # # Fix typos
    # cleaned_data["Breed"] = cleaned_data["Breed"].replace ({
    #                                 "Holstien": "Holstein",
    #                                 " Brown Swiss": "Brown Swiss",
    #                                 "Brown Swiss ": "Brown Swiss"})
    
    # Further processing for training set
    if is_train or LABEL_COL in cleaned_data:
        # Remove noise
        cleaned_data = cleaned_data[cleaned_data[LABEL_COL] >= 0]

        # Get labels
        labels = cleaned_data[LABEL_COL].values.ravel ()
        cleaned_data = cleaned_data.drop (LABEL_COL, axis = 1)

    # Return
    return labels, cleaned_data


def engineer_data (
    cleaned_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Feature Engineering
    Returns engineered dataframe
    """
    # Convert date to season
    months = pd.to_datetime (cleaned_data['Date']).dt.month
    engineered_data = cleaned_data.drop (columns = ['Date'])
    
    # split up months
    def month_to_season (m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
        
    engineered_data['Season'] = months.apply (month_to_season)
    if ('Date' in CAT_FEATURES.keys ()):
        CAT_FEATURES.pop ('Date')
    CAT_FEATURES['Season'] = RecordType.KEEP

    # Impute
    impute_fts = [col for col in engineered_data.columns 
                  if engineered_data[col].isna ().sum () > 0]
    for col in impute_fts:
        engineered_data[col] = engineered_data[col].fillna (
                                                engineered_data[col].median ())

    # One Hot encode
    onehot_fts = [ft for ft, t in CAT_FEATURES.items () if t == RecordType.KEEP]
    engineered_data = pd.get_dummies (engineered_data,
                                      columns = onehot_fts,
                                      drop_first = True)

    # Drop records w/ missing cols
    engineered_data = engineered_data.dropna ()

    # Flag extreme values
    engineered_data['Water_Intake_L'] = engineered_data['Water_Intake_L'].clip (
                                                        lower = 30, upper=  140)

    return engineered_data


if __name__ == '__main__':
    pass