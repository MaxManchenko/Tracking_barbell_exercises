import json
import pandas as pd
import numpy as np

from src.utils.OutlierDetectors import mark_outliers_IQR

config_path = "config.json"


# --------------------------------------------------------------
# Load configuration settings and load the dataset
# --------------------------------------------------------------
with open(config_path, "r") as config_file:
    config = json.load(config_file)

dataset_path = config["data_path_01_dataset"]
path_to_save_data = config["data_path_02_outliers_removed"]

df = pd.read_pickle(dataset_path)
outlier_columns = df.columns[:6].tolist()

# --------------------------------------------------------------
# Dealing with outliers with IQR
# --------------------------------------------------------------
outliers_removed_df = df.copy()

for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_IQR(df[df["label"] == label], col=col)

        # Replace values marked as outliers with NaN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # Update the outliers_remobed_df
        outliers_removed_df.loc[(outliers_removed_df["label"] == label, col)] = dataset[
            col
        ]

        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} outliers from {col} for {label}")

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_pickle(path_to_save_data)
