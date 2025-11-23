import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset csv file
df = pd.read_csv('data/in/cattle_data_train.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear

# Define seasons based on month
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Date'].dt.month.apply(assign_season)

# Create directory to save plots if it doesn't exist
output_dir = 'plots_by_feature'
os.makedirs(output_dir, exist_ok=True)

# List columns to skip (non-numeric or non-plot relevant)
skip_columns = ['Cattle_ID', 'Breed', 'Climate_Zone', 'Management_System', 'Date', 'Farm_ID', 
                'FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine', 'BQ_Vaccine', 
                'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine', 'Rabies_Vaccine', 'Mastitis']

# Loop through columns and plot each numeric feature
for col in df.columns:
    if col not in skip_columns and pd.api.types.is_numeric_dtype(df[col]):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x='DayOfYear', y=col, hue='Season', palette='Set2')
        plt.title(f'{col} vs Day of Year with Season Overlay')
        plt.xlabel('Day of Year')
        plt.ylabel(col)
        plt.legend(title='Season')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_vs_DayOfYear.png")
        plt.close()
