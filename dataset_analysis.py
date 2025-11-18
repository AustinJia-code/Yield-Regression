import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
import os



train_path = 'data/in/cattle_data_train.csv'
test_path = 'data/in/cattle_data_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# Keep only common columns
common_cols = [col for col in train.columns if col in test.columns]
train = train[common_cols]
test = test[common_cols]

# Separate numeric and categorical
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train.select_dtypes(include=['object', 'category']).columns

print("\nNumeric columns:", list(numeric_cols))
print("Categorical columns:", list(categorical_cols))

os.makedirs("distribution_plots", exist_ok=True)

print("\n=== NUMERIC FEATURE DISTRIBUTION COMPARISON ===")
for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.kdeplot(train[col], label="Train", fill=True)
    sns.kdeplot(test[col], label="Test", fill=True)
    plt.title(f"Distribution of {col} (Train vs Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"distribution_plots/{col}_numeric.png")
    plt.close()

    # KS-Test
    train_col = train[col].dropna()
    test_col = test[col].dropna()
    ks_p = ks_2samp(train_col, test_col).pvalue
    print(f"{col}: KS p-value = {ks_p:.4f}")

print("\n=== CATEGORICAL FEATURE DISTRIBUTION COMPARISON ===")
for col in categorical_cols:
    plt.figure(figsize=(7,4))
    
    try:
        train_counts = train[col].value_counts(normalize=True)
        test_counts = test[col].value_counts(normalize=True)

        # Plot
        comp_df = pd.DataFrame({
            "Train": train_counts,
            "Test": test_counts
        }).fillna(0)

        comp_df.plot(kind="bar", figsize=(8,4))
        plt.title(f"Category Distribution: {col}")
        plt.tight_layout()
        plt.savefig(f"distribution_plots/{col}_categorical.png")
        plt.close()

        # Chi-square test
        contingency = pd.concat([train_counts, test_counts], axis=1).fillna(0)
        chi_p = chi2_contingency(contingency)[1]
        print(f"{col}: Chi-square p-value = {chi_p:.4f}")

    except:
        print(f"Skipped categorical feature (invalid data): {col}")


print("\nAll plots saved in folder: distribution_plots/")
print("Done!")
