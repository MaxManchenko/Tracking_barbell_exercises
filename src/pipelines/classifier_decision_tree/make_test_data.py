import json
import pandas as pd

from src.pipelines.classifier_decision_tree.DataProcessingPipeline import (
    DataProcessingPipeline,
)

data_config_path = "configs/data_config_classifier.json"


# Load configuration settings and build the path to the files
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

data_path_out = data_config["data_paths"].get("X_y_test")

data_path_X_test = data_config["data_paths"].get("X_test")
data_path_y_test = data_config["data_paths"].get("y_test")


# Run data processing pipeline
def main():
    pipeline = DataProcessingPipeline(data_config_path, test=True)
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
