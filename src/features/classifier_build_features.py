import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.utils.DataTransformation import LowPassFilter
from src.utils.TemporalAbstraction import NumericalAbstraction
from src.utils.FrequencyAbstraction import FourierTransformation

config_path = "config.json"


# --------------------------------------------------------------
# Load configuration settings and build the path to the files
# --------------------------------------------------------------
with open(config_path, "r") as config_file:
    config = json.load(config_file)

dataset_path = config["data_path_02_outliers_removed"]
path_to_save_data = config["data_path_03_features"]

df = pd.read_pickle(dataset_path)
predictor_columns = df.columns[:6].tolist()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()

# Sampling_frequency
# (we set frequency = 200 ms when we did the resampling in src/data/make_dataset.py)
fs = 1000 / 200
cutoff = 1.3

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass.drop([col + "_lowpass"], axis=1, inplace=True)

# ----------------------------- ---------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
# ELBOW TECHNIQUE
# Finding the optimal component number by plotting the variance captured against
# the component number and then selecting the point at which the rate of change
# in variance diminishes (the "elbow"), as this is typically the point at which
# adding more components does not significantly improve the analysis.

df_pca = df_lowpass.copy()

pca = PCA(n_components=4)
df_pca_transformed_np = pca.fit_transform(df_pca[predictor_columns].to_numpy())
df_pca_transformed_pd = pd.DataFrame(
    df_pca_transformed_np,
    index=df_pca.index,
    columns=["pca_1", "pca_2", "pca_3", "pca_4"],
)
df_pca = pd.concat([df_pca, df_pca_transformed_pd], axis=1)

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = (
    df_squared["acc_x"].pow(2) + df_squared["acc_y"].pow(2) + df_squared["acc_z"].pow(2)
)
gyr_r = (
    df_squared["gyr_x"].pow(2) + df_squared["gyr_y"].pow(2) + df_squared["gyr_z"].pow(2)
)

df_squared["acc_r"] = acc_r.pow(0.5)
df_squared["gyr_r"] = gyr_r.pow(0.5)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)  # window size

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal.query("set == @s").copy()
    subset = NumAbs.abstract_numerical(
        subset, predictor_columns, window_size=ws, aggregation_function="mean"
    )
    subset = NumAbs.abstract_numerical(
        subset, predictor_columns, window_size=ws, aggregation_function="std"
    )
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# --------------------------------------------------------------
# Frequency features (Fourier transformation)
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sr = int(1000 / 200)  # sampling rate (the number of samples per second)
ws = 14

df_freq_list = []
for s in tqdm(df_freq["set"].unique(), desc="Processing sets"):
    subset = df_freq.query("set == @s").copy().reset_index(drop=True)
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, sr)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq.dropna(inplace=True)
df_freq = df_freq.iloc[::2]  # select every second row

# df_freq copy for experiments with KMeans
# df_freq.to_pickle("../../data/interim/for_experiments_with_KMeans.pkl")

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
subset = df_cluster[cluster_columns]
k = 5  # number of clusters


kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(subset)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle(path_to_save_data)
