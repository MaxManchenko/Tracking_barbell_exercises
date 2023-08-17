import json
from glob import glob

from src.pipelines.classifier_decision_tree.MakePreProcessedData import (
    DataPreProcessor,
)
from src.pipelines.classifier_decision_tree.MakeProcessedData import DataProcessor
from src.pipelines.classifier_decision_tree.MakeFeatures import make_features


class DataProcessingPipeline:
    """
    A class representing a data processing pipeline for preparing raw sensor data
    (stored in separate files) for input into a ML model: "files in -> data out".

    This pipeline includes data loading, preprocessing, feature extraction, and
    data saving steps.

    Args:
        data_config_path (str): Path to the JSON data configuration file.

    Methods:
        load_config():
            Load pipeline settings from the specified configuration file.
        load_data():
            Load and resample raw sensor data based on the configuration settings.
        preprocess_data():
            Perform data preprocessing, including outlier removal and imputation of missing data.
        process_data():
            Apply further processing steps, such as filtering and attribute transformations.
        extract_features():
            Extract relevant features from the processed data.
        save_processed_data():
            Save the processed data to a specified location.
    """

    def __init__(self, data_config_path, test=False):
        """
        Initialize the DataProcessingPipeline instance.

        Args:
            data_config_path (str): Path to the JSON data configuration file.
        """
        self.data_config = data_config_path
        self.test = test
        self.load_config()

    def load_config(self):
        """
        Load data configuration settings from the JSON file.
        """
        if self.test:
            try:
                with open(self.data_config, "r") as data_config_file:
                    self.data_config = json.load(data_config_file)
                    self.files_path_in = self.data_config["data_paths"].get("test_raw")
                    self.data_path_out = self.data_config["data_paths"].get("X_y_test")
                    self.file_pattern = self.data_config.get("file_pattern", "csv")
            except FileExistsError:
                print("Data config file not found.")
                self.files_path_in = None
                self.data_path_out = None
        else:
            try:
                with open(self.data_config, "r") as data_config_file:
                    self.data_config = json.load(data_config_file)
                    self.files_path_in = self.data_config.get("files_path_in")
                    self.data_path_out = self.data_config["data_paths"].get(
                        "fully_processed_out"
                    )
                    self.file_pattern = self.data_config.get("file_pattern", "csv")
            except FileExistsError:
                print("Data config file not found.")
                self.files_path_in = None
                self.data_path_out = None

    def run(self):
        """
        Execute the data processing pipeline.
        """
        self.load_data()
        self.preprocess_data()
        self.process_data()
        self.extract_features()
        self.save_processed_data()

    def load_data(self):
        """
        Load the input dataset.
        """
        full_path_pattern = self.files_path_in + "*" + self.file_pattern
        self.files = glob(full_path_pattern)

    def preprocess_data(self):
        """
        Perform data preprocessing.
        """
        data_preprocessor = DataPreProcessor()
        # Read data
        self.acc_df, self.gyr_df = data_preprocessor.read_data_from_files(self.files)
        # Merge data
        self.data_merged = data_preprocessor.merge_dataframes(self.acc_df, self.gyr_df)
        # Resample data
        self.data_resampled = data_preprocessor.resample_data(self.data_merged)

    def process_data(self):
        """
        Perform data processing.
        """
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

    def extract_features(self):
        """
        Build the final dataset with selected features.
        """
        self.df_selected = make_features(self.data_freq)

    def save_processed_data(self):
        """
        Save the processed data to an output file.
        """
        self.df_selected.to_pickle(self.data_path_out)


if __name__ == "__main__":
    pipeline = DataProcessingPipeline()
    pipeline.run()
