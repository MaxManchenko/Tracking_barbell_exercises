import json
import pandas as pd

from src.pipelines.classifier_decision_tree.DataProcessingPipeline import (
    DataProcessingPipeline,
)

config_path = "config.json"


# Load configuration settings and build the path to the files
with open(config_path, "r") as config_file:
    config = json.load(config_file)

files_path_in = config["data_path_test_raw"]
data_path_out = config["data_path_X_y_test"]

data_path_X_y_test = config["data_path_X_y_test"]
data_path_X_test = config["data_path_X_test"]
data_path_y_test = config["data_path_y_test"]


# Run data processing pipeline
def main():
    pipeline = DataProcessingPipeline(
        config_path, files_path_in=files_path_in, data_path_out=data_path_out
    )
    pipeline.run()


if __name__ == "__main__":
    main()

# Load test data
df = pd.read_pickle(data_path_out)
X_test = df.drop("label", axis=1)
y_test = df["label"]

# Save X, y test data
X_test.to_pickle(data_path_X_test)
y_test.to_pickle(data_path_y_test)
