import json

from src.pipelines.classifier_decision_tree.DataProcessingPipeline import (
    DataProcessingPipeline,
)

data_config_path = "configs/data_config_classifier.json"


# Run data processing pipeline
def main():
    pipeline = DataProcessingPipeline(data_config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
