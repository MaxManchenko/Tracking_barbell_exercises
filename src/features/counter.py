import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema

pd.options.mode.chained_assignment = None

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_precessed.pkl")
df = df.query('label!="rest"')

acc_r = df["acc_x"].pow(2) + df["acc_y"].pow(2) + df["acc_z"].pow(2)
gyr_r = df["gyr_x"].pow(2) + df["gyr_y"].pow(2) + df["gyr_z"].pow(2)
df["acc_r"] = acc_r.pow(0.5)
df["gyr_r"] = gyr_r.pow(0.5)

# --------------------------------------------------------------
# Split data by the exercise
# --------------------------------------------------------------
bench_df = df.query('label =="bench"')
squat_df = df.query('label =="squat"')
row_df = df.query('label =="row"')
ohp_df = df.query('label =="ohp"')
dead_df = df.query('label =="dead"')

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
# Sampling_frequency
# (we set frequency = 200 ms when we did the resampling in src/data/make_dataset.py)
fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, column="acc_r", fs=5.0, cutoff=0.4, order=10):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indixes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indixes]

    return len(peaks)


count_reps(bench_set, cutoff=0.4)
