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


# class CountRepetitions:
#     def __init__(self, data_config_path, model_config_path):
#         self.data_config = data_config_path
#         self.model_config = model_config_path
#         self.load_data_config()
#         self.load_model_config()

#     def load_data_config(self):
#         with open(self.data_config_path, "r") as data_config_file:
#             self.data_config = json.load(data_config_file)
#             self.data_path_in = self.data_config["data_path_out"]

#     def load_model_config(self):
#         with open(self.model_config_path, "r") as model_config_file:
#             self.model_config = json.load(model_config_file)
#             self.column = self.model_config["column"]
#             self.column_row = self.model_config["column_row"]
#             self.cutoff = self.model_config["cutoff"]
#             self.cutoff_squat = self.model_config["cutoff_squat"]
#             self.cutoff_row = self.model_config["cutoff_row"]
#             self.cutoff_ohp = self.model_config["cutoff_ohp"]
#             self.fs = self.model_config["fs"]
#             self.order = self.model_config["order"]

#     def load_data(self):
#         self.df = pd.read_pickle(self.data_path_in)

#     def process_data(self):
#         LowPass = LowPassFilter()
#         for s in self.df["set"].unique():
#             self.subset = self.df[self.df["set"] == s]

#             if self.subset["label"].iloc[0] == "suqat":
#                 self.cutoff = 0.35

#             elif self.subset["label"].iloc[0] == "row":
#                 self.cutoff = 0.65
#                 self.column = "gyr_x"

#             elif self.subset["label"].iloc[0] == "ohp":
#                 self.cutoff = 0.35

#             self.data_lowpass = LowPass.low_pass_filter(
#                 self.df,
#                 col=self.column,
#                 sampling_frequency=self.fs,
#                 cutoff_frequency=self.cutoff,
#                 order=self.order,
#             )

#     def count_reps(self):
#         indixes = argrelextrema(
#             self.data_lowpass[self.column + "_lowpass"].values, np.greater
#         )
#         peaks = self.data_lowpass.iloc[indixes]

#         result = len(peaks)

#         return result


class CountRepetitions:
    def __init__(self, data_config_path, model_config_path):
        self.data_config_path = data_config_path
        self.model_config_path = model_config_path
        self.load_data_config()
        self.load_model_config()

    def load_data_config(self):
        try:
            with open(self.data_config_path, "r") as data_config_file:
                self.data_config = json.load(data_config_file)
                self.data_path_in = self.data_config.get("data_path_out")
        except FileNotFoundError:
            print("Data config file not found.")
            self.data_path_in = None

    def load_model_config(self):
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
        try:
            self.df = pd.read_pickle(self.data_path_in)
        except FileNotFoundError:
            print("Dataset not found.")
            self.df = None

    def process_data(self):
        LowPass = LowPassFilter()

        label = self.df["label"].iloc[0]
        self.cutoff = self.cutoff_dict.get(label, self.cutoff_dict.get("default", 0.4))
        self.column = self.column_row if label == "row" else self.column

        self.data_lowpass = LowPass.low_pass_filter(
            self.df,
            col=self.column,
            sampling_frequency=self.fs,
            cutoff_frequency=self.cutoff,
            order=self.order,
        )

    def count_reps(self):
        indixes = argrelextrema(
            self.data_lowpass[self.column + "_lowpass"].values, np.greater
        )
        peaks = self.data_lowpass.iloc[indixes]

        result = len(peaks)

        return result


# Example usage

counter = CountRepetitions(data_config_path, model_config_path)
counter.load_data()
counter.process_data()
repetition_count = counter.count_reps()
print("Repetition count:", repetition_count)
