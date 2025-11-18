import pandas as pd

df = pd.read_csv("data/in/cattle_data_train.csv")

# All columns EXCEPT Milk_Yield_L
key_columns = [col for col in df.columns if col != "Milk_Yield_L"]

# Find duplicate groups with differing milk yields
duplicate_groups = (
    df.groupby(key_columns)["Milk_Yield_L"]
      .nunique()
      .reset_index()
)

# Keep only groups where yield is inconsistent
inconsistent = duplicate_groups[duplicate_groups["Milk_Yield_L"] > 1]

print("Number of inconsistent duplicate records:", len(inconsistent))

if not inconsistent.empty:
    # merge back to original df to show full rows
    inconsistent_rows = df.merge(inconsistent[key_columns], on=key_columns, how="inner")
    print("\nInconsistent rows:")
    print(inconsistent_rows)
