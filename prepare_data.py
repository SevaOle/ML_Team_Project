import numpy as np
import pandas as pd


INPUT_PATH = "data/dataset.csv"
OUTPUT_PATH = "data/cleaned_dataset.csv"
LABEL_COL = "Label"


df = pd.read_csv(INPUT_PATH)

# Clean column names because many CICIDS-style datasets have spaces before names
df.columns = df.columns.str.strip()

# Remove exact duplicate rows first
df = df.drop_duplicates()

# Drop columns that can cause leakage / memorization
df = df.drop(columns=[
    "Flow ID",          # we dont need an ID
    "Source IP",        # can memorize
    "Destination IP",   # can memorize
    "Timestamp",        # may memorize attack time
    "Source Port",      # usually random, can memorize
    "Destination Port"  # often port 80 for DDOS -> can memorize and classify legitimate port 80 traffic as attacks
], errors="ignore")


# Drop columns that have only one value
constant_cols = [
    col for col in df.columns
    if col != LABEL_COL and df[col].nunique(dropna=False) <= 1
]

df = df.drop(columns=constant_cols, errors="ignore")


# Drop duplicate columns by content
# Keeping these rare flags for now because they may still carry useful info
rare_flag_cols = [
    "FIN Flag Count",
    "RST Flag Count",
    "ECE Flag Count"
]

duplicate_content_cols = df.columns[df.T.duplicated()].tolist()

duplicate_content_cols = [
    col for col in duplicate_content_cols
    if col not in rare_flag_cols
]

df = df.drop(columns=duplicate_content_cols, errors="ignore")


# Replace infinite values with NaN, then fill numeric missing values with median
df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remove rows without a label, just in case
df = df.dropna(subset=[LABEL_COL])

# Remove duplicates again after dropping columns
df = df.drop_duplicates()


df.to_csv(OUTPUT_PATH, index=False)

print("Cleaned dataset saved to:", OUTPUT_PATH)
print("Final shape:", df.shape)
print("\nLabel counts:")
print(df[LABEL_COL].value_counts())