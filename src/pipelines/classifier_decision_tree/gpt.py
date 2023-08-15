import json
from glob import glob

from src.pipelines.classifier_decision_tree.make_preprocessed_data import (
    DataPreProcessor,
)
from src.pipelines.classifier_decision_tree.make_processed_data import DataProcessor
from src.pipelines.classifier_decision_tree.make_features import make_features


class DataProcessingPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.process_data()
        self.extract_features()
        self.save_processed_data()

    def load_data(self):
        # Load config. settings and Build the path to the .csv files
        files_path = self.config["files_path"]
        file_pattern = self.config["file_pattern"]
        full_path_pattern = files_path + "*" + file_pattern
        self.files = glob(full_path_pattern)

    def preprocess_data(self):
        # Call DataPreProcessor
        data_preprocessor = DataPreProcessor()
        # Read data
        self.acc_df, self.gyr_df = data_preprocessor.read_data_from_files(self.files)
        # Merge data
        self.data_merged = data_preprocessor.merge_dataframes(self.acc_df, self.gyr_df)
        # Resample data
        self.data_resampled = data_preprocessor.resample_data(self.data_merged)

    def process_data(self):
        # Call DataProcessor
        data_processor = DataProcessor()
        # Mark outliers
        self.data_w_marked_outliers = data_processor.remove_outliers_by_IQR(
            self.data_resampled
        )
        # Impute missing values
        self.data_w_missing_data_filled_in = data_processor.impute_missing_values(
            self.data_w_marked_outliers
        )
        # Calculate set duration
        self.data_w_set_duration = data_processor.calculate_set_duration(
            self.data_w_missing_data_filled_in
        )
        # Apply lowpass filter
        self.data_lowpass = data_processor.lowpass_filter(self.data_w_set_duration)

        # Add square attributes
        self.data_w_square_attr = data_processor.square_attributes(self.data_lowpass)

        # Calculate rolling average
        self.data_temporal = data_processor.rolling_averege(self.data_w_square_attr)

        # Make Fourier transformation
        self.data_freq = data_processor.fourier_transform(self.data_temporal)

    # Build the final dataset with selected features
    def extract_features(self):
        self.df_selected = make_features(self.data_freq)

    # Save the fully processed data
    def save_processed_data(self):
        path_to_save_processed_data = self.config["data_path_fully_processed"]
        self.df_selected.to_pickle(path_to_save_processed_data)


if __name__ == "__main__":
    config_path = "config.json"
    pipeline = DataProcessingPipeline(config_path)
    pipeline.run()
