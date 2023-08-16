import numpy as np
import pandas as pd

from src.utils.OutlierDetectors import mark_outliers_IQR
from src.utils.FrequencyAbstraction import FourierTransformation
from src.utils.TemporalAbstraction import NumericalAbstraction
from src.utils.DataTransformation import LowPassFilter


class DataProcessor:
    def remove_outliers_by_IQR(self, df):
        """Remove outliers using Interquartile range technique (IQR).

        Args:
            df (pd.DataFrame): Preprocessed data.

        Returns:
            pd.DataFrame: The dataframe with values
            marked as outliers with NaN.
        """

        outlier_columns = df.columns[:6].tolist()

        # Dealing with outliers with IQR
        outliers_removed_df = df.copy()

        for col in outlier_columns:
            for label in df["label"].unique():
                dataset = mark_outliers_IQR(df[df["label"] == label], col=col)

                # Replace values marked as outliers with NaN
                dataset.loc[dataset[col + "_outlier"], col] = np.nan

                # Update the outliers_removed_df
                outliers_removed_df.loc[
                    (outliers_removed_df["label"] == label, col)
                ] = dataset[col]

        df.drop(index=df.index, columns=df.columns, inplace=True)

        return outliers_removed_df

    def impute_missing_values(self, df):
        """Dealing with missing values by interpolation.

        Args:
            df (pd.DataFrame): Dataframe with missing values.

        Returns:
            pd.DataFrame: Dataframe with filled in missing values.
        """

        predictor_columns = df.columns[:6].tolist()
        for col in predictor_columns:
            df[col] = df[col].interpolate()

        return df

    def calculate_set_duration(self, df):
        """Calculate the duration of the set in seconds and add the values
        to the new "duration" column.

        Args:
            df (pd.DataFrame): Dataframe with missing values filled in.

        Returns:
            pd.DataFrame: Dataframe with the new "duration" column.
        """

        for s in df["set"].unique():
            start = df[df["set"] == s].index[0]
            stop = df[df["set"] == s].index[-1]
            duration = stop - start
            df.loc[(df["set"] == s), "duration"] = duration.seconds

        return df

    def lowpass_filter(self, df):
        """Apply Butterworth lowpass filter to make data less noisy.

        Args:
            df (pd.DataFrame): Dataframe with no missing values.

        Returns:
            pd.DataFrame: Dataframe with smoother date.
        """

        predictor_columns = df.columns[:6].tolist()

        df_lowpass = df.copy()
        LowPass = LowPassFilter()

        # Sampling_frequency
        # (we set frequency = 200 ms)
        fs = 1000 / 200
        cutoff = 1.3

        for col in predictor_columns:
            df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
            df_lowpass[col] = df_lowpass[col + "_lowpass"]
            df_lowpass.drop([col + "_lowpass"], axis=1, inplace=True)

        df.drop(index=df.index, columns=df.columns, inplace=True)

        return df_lowpass

    def square_attributes(self, df):
        """Build the sum of squares of the acceleration and gyroscope components
        along the x, y, z axises that are invariant to device orientation

        Args:
            df (pd.DataFrame): _description_
        """

        df_squared = df.copy()

        acc_r = (
            df_squared["acc_x"].pow(2)
            + df_squared["acc_y"].pow(2)
            + df_squared["acc_z"].pow(2)
        )
        gyr_r = (
            df_squared["gyr_x"].pow(2)
            + df_squared["gyr_y"].pow(2)
            + df_squared["gyr_z"].pow(2)
        )

        df_squared["acc_r"] = acc_r.pow(0.5)
        df_squared["gyr_r"] = gyr_r.pow(0.5)

        df.drop(index=df.index, columns=df.columns, inplace=True)

        return df_squared

    def rolling_averege(self, df):
        """Calculate rolling average (mean) to smooth out small fluctuations
        in the dataset, while gaining insight into trends.

        Args:
            df (pd.DataFrame): Dataframe

        Returns:
            pd.DataFrame: Dataframe
        """

        # Temporal abstraction
        df_temporal = df.copy()
        NumAbs = NumericalAbstraction()

        predictor_columns = df.columns[:6].tolist() + ["acc_r", "gyr_r"]

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

        df.drop(index=df.index, columns=df.columns, inplace=True)

        return df_temporal

    def fourier_transform(self, df):
        """Apply discrete Fourier transfomation

        Args:
            df (pd.DataFrame): Dataframe

        Returns:
            pd.DataFrame: Dataframe
        """

        df_freq = df.copy().reset_index()
        FreqAbs = FourierTransformation()

        predictor_columns = df.columns[:6].tolist() + ["acc_r", "gyr_r"]
        sr = int(1000 / 200)  # sampling rate (the number of samples per second)
        ws = 14

        df_freq_list = []
        for s in df_freq["set"].unique():
            subset = df_freq.query("set == @s").copy().reset_index(drop=True)
            subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, sr)
            df_freq_list.append(subset)

        df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

        # Dealing with overlapping windows
        df_freq.dropna(inplace=True)
        df_freq = df_freq.iloc[::2]  # select every second row

        df.drop(index=df.index, columns=df.columns, inplace=True)

        return df_freq
