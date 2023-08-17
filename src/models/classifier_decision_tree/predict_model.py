import joblib
import json
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

data_config_path = "configs/data_config_classifier.json"


# Load config. settings and Build the path to the files
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

model_path = data_config["model_path"]
X_test_path = data_config["data_paths"].get("X_test")
y_test_path = data_config["data_paths"].get("y_test")

# Load the trained model
classifier_dt = joblib.load(model_path)

# Load the test dataset
X_test = pd.read_pickle(X_test_path)
y_test = pd.read_pickle(y_test_path)

# Perform predictions
predictions = classifier_dt.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
