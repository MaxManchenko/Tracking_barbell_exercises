import json
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from src.utils.DataTransformation import LowPassFilter
from src.pipelines.counter.DataProcessingPipeline import DataProcessingPipelineCounter

pd.options.mode.chained_assignment = None


# --------------------------------------------------------------
# Config paths for the data and model
# --------------------------------------------------------------
data_config_path = "configs/data_config_counter.json"
model_config_path = "configs/model_config_counter.json"


# --------------------------------------------------------------
# Run data processing pipeline
# --------------------------------------------------------------
def main():
    pipeline = DataProcessingPipelineCounter(data_config_path)
    pipeline.run()


if __name__ == "__main__":
    main()


# --------------------------------------------------------------
# Make class to count repetitions
# --------------------------------------------------------------
class CountRepetitions:
    """
    A class for counting repetitions in each set.

    This class provides methods to load sensor data, process the data to identify
    repetitions, and perform counting for different types of exercises.

    Attributes:
        data_config_path (str): Path to the data configuration JSON file.

    Methods:
        load_data_config(): Load data configuration from JSON file.
        load_data(): Load sensor data from a pickle file.
        process_data(): Process sensor data to prepare for repetition counting.
        count_reps_one_set(subset: pd.DataFrame): Count repetitions for a specific exercise set.
        count_reps(): Count repetitions for all exercise sets.
    """

    def __init__(self, data_config_path, model_config_path):
        """
        Initialize the CountRepetitions instance.

        Args:
            data_config_path (str): Path to the data configuration JSON file.
        """
        self.data_config_path = data_config_path
        self.model_config_path = model_config_path
        self.load_data_config()
        self.load_model_config()
        self.load_data()
        self.process_data()

    def load_data_config(self):
        """
        Load data configuration from JSON file.
        """
        try:
            with open(self.data_config_path, "r") as data_config_file:
                self.data_config = json.load(data_config_file)
                self.data_path_in = self.data_config.get("data_path_out")
        except FileNotFoundError:
            print("Data config file not found.")
            self.data_path_in = None

    def load_model_config(self):
        """
        Load model configuration from JSON file.
        """
        try:
            with open(self.model_config_path, "r") as model_config_file:
                self.model_config = json.load(model_config_file)
                self.column = self.model_config.get("column")
                self.column_row = self.model_config.get("column_row")
                self.fs = self.model_config.get("fs")
                self.order = self.model_config.get("order")
                self.cutoff_dict = self.model_config.get("cutoff_dict", {})
        except FileNotFoundError:
            print("Model config file not found.")
            self.column = None
            self.column_row = None
            self.fs = None
            self.order = None
            self.cutoff_dict = {}

    def load_data(self):
        """
        Load data from a pickle file.
        """
        try:
            self.df = pd.read_pickle(self.data_path_in)
        except FileNotFoundError:
            print("Dataset not found.")
            self.df = None

    def process_data(self):
        """
        Process data to prepare for repetition counting.
        """
        LowPass = LowPassFilter()

        label = self.df["label"].iloc[0]
        self.cutoff = self.cutoff_dict.get(label, self.cutoff_dict.get("default", 0.4))
        self.column = self.column_row if label == "row" else self.column

        self.df_lowpass = LowPass.low_pass_filter(
            self.df,
            col=self.column,
            sampling_frequency=self.fs,
            cutoff_frequency=self.cutoff,
            order=self.order,
        )

    def count_reps_one_set(self, subset):
        """
        Count repetitions for a specific exercise set.

        Args:
            subset (pd.DataFrame): Subset of data for a specific exercise set.

        Returns:
            int: Count of repetitions.
        """
        indixes = argrelextrema(subset[self.column + "_lowpass"].values, np.greater)
        peaks = subset.iloc[indixes]
        counts = len(peaks)
        return int(counts)

    def count_reps(self):
        """
        Count repetitions for all exercise sets.

        Returns:
            pd.DataFrame: DataFrame with repetitions counted for each exercise set.
        """
        self.df_lowpass["reps_pred"] = 0
        for s in self.df_lowpass["set"].unique():
            subset = self.df_lowpass.query("set == @s")
            reps = self.count_reps_one_set(subset)
            self.df_lowpass.loc[self.df["set"] == s, "reps_pred"] = reps

        rep_df = (
            self.df_lowpass.groupby(["label", "category", "set"])["reps_pred"]
            .max()
            .reset_index()
        )
        return rep_df


# --------------------------------------------------------------
# Verification
# --------------------------------------------------------------
counter = CountRepetitions(data_config_path, model_config_path)
repetition_count = counter.count_reps()
