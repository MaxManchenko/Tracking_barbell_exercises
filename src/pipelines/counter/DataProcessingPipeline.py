import json
import pandas as pd

pd.options.mode.chained_assignment = None


class DataProcessingPipelineCounter:
    """
    A class for processing data and generating sum of squares of acceleration
    and gyroscope components along the x, y, z axes that are invariant to device orientation.

    Attributes:
        data_config_path (str): Path to the JSON data configuration file.

    Methods:
        load_data_config(): Load data configuration settings from the JSON file.
        run(): Execute the data processing pipeline.
        load_data(): Load the input dataset.
        process_data(): Perform data processing by calculating the sum of squares.
        save_processed_data(): Save the processed data to an output file.
    """

    def __init__(self, data_config_path):
        """
        Initialize the DataProcessingPipelineCounter instance.

        Args:
            data_config_path (str): Path to the JSON data configuration file.
        """
        self.data_config = data_config_path
        self.load_data_config()

    def load_data_config(self):
        """
        Load data configuration settings from the JSON file.
        """
        try:
            with open(self.data_config, "r") as data_config_file:
                self.data_config = json.load(data_config_file)
                self.data_path_in = self.data_config["data_path_in"]
                self.data_path_out = self.data_config["data_path_out"]
        except FileNotFoundError:
            print("Data config file not found.")
            self.data_path_in = None
            self.data_path_out = None

    def run(self):
        """
        Execute the data processing pipeline.
        """
        self.load_data()
        if self.df is not None:
            self.process_data()
            self.save_processed_data()

    def load_data(self):
        """
        Load the input dataset.
        """
        try:
            self.df = pd.read_pickle(self.data_path_in)
        except FileNotFoundError:
            print("Dataset not found.")
            self.df = None

    def process_data(self):
        """
        Perform data processing by calculating the sum of squares.
        """
        self.df = self.df.query('label!="rest"')
        self.acc_r = (
            self.df["acc_x"].pow(2) + self.df["acc_y"].pow(2) + self.df["acc_z"].pow(2)
        )
        self.df["acc_r"] = self.acc_r.pow(0.5)

    def save_processed_data(self):
        """
        Save the processed data to an output file.
        """
        if self.data_path_out is not None:
            self.df.to_pickle(self.data_path_out)


if __name__ == "__main__":
    data_config_path = "configs/data_config_counter.json"
    pipeline = DataProcessingPipelineCounter(data_config_path)
    pipeline.run()
