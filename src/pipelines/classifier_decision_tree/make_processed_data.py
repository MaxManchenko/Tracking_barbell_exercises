import json

from src.pipelines.classifier_decision_tree.DataProcessingPipeline import (
    DataProcessingPipeline,
)

config_path = "configs/data_config_classifier.json"


# Load configuration settings and build the path to the files
with open(config_path, "r") as config_file:
    config = json.load(config_file)

files_path_in = config["files_path_in"]
data_path_out = config["data_path_fully_processed_out"]


# Run data processing pipeline
def main():
    pipeline = DataProcessingPipeline(
        config_path, files_path_in=files_path_in, data_path_out=data_path_out
    )
    pipeline.run()


if __name__ == "__main__":
    main()
