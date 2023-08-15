import pandas as pd


class DataPreProcessor:
    """Create a pandas dataframe for further outliers remouval and
    feature engineering.
    """

    def read_data_from_files(self, files):
        """Build two pandas dataframes for accelerometer and gyroscope data.

        Args:
            files (list): A list of .csv files.

        Returns:
            pd.DataFrame: Two dataframes "acc_df" and "gyr_df" for
            accelerometer and gyroscope data.
        """

        acc_df = pd.DataFrame()
        gyr_df = pd.DataFrame()

        acc_set = 1
        gyr_set = 1

        for f in files:
            participant = f.split("-")[0][-1]
            label = f.split("-")[1]
            category = f.split("-")[2].split("_")[0].rstrip("123")

            df = pd.read_csv(f)
            df["participant"] = participant
            df["label"] = label
            df["category"] = category

            if "Accelerometer" in f:
                df["set"] = acc_set
                acc_set += 1
                acc_df = pd.concat([acc_df, df])

            if "Gyroscope" in f:
                df["set"] = gyr_set
                gyr_set += 1
                gyr_df = pd.concat([gyr_df, df])

        acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

        acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
        gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

        return acc_df, gyr_df

    def merge_dataframes(self, acc_df, gyr_df):
        """Merge two dataframes with accelerometer and gyroscope data
        into one dataframe.

        Args:
            acc_df (pd.DataFrame): Accelerometer data
            gyr_df (pd.DataFrame): Gyroscope data

        Returns:
            pd.DataFrame: The merged dataset with accelerometer
            and gyroscope data.
        """

        # Merging datasets
        pd.concat([acc_df, gyr_df], axis=1)

        # Rename columns
        data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
        data_merged.columns = [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
            "participant",
            "label",
            "category",
            "set",
        ]

        return data_merged

    def resample_data(self, data_merged):
        """Goup the data by day and then downsample as follows:
        1) Downsample the frequency series into 200 ms bins
        2) The accelerometer and gyroscope values to mean values
        3) Get the last values for "participant", "labes", "category", "set" columns

        Args:
            data_merged (pd.DataFrame): Merged data

        Returns:
            pd.DataFrame: Resampled data
        """

        # Accelerometer:    12.500HZ
        # Gyroscope:        25.000Hz

        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
            "participant": "last",
            "label": "last",
            "category": "last",
            "set": "last",
        }

        # Split by day and downsample
        days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
        data_resampled = pd.concat(
            df.resample(rule="200ms").apply(sampling).dropna() for df in days
        )

        data_resampled["set"] = data_resampled["set"].astype("int")

        return data_resampled
