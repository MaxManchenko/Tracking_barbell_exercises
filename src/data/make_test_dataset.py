from glob import glob
import json

config_path = "config.json"


# --------------------------------------------------------------
# Load configuration settings and build the path to the files
# --------------------------------------------------------------
with open(config_path, "r") as config_file:
    config = json.load(config_file)

files_path = config["files_path"]
file_pattern = config["file_pattern"]
full_path_pattern = files_path + "*" + file_pattern

# --------------------------------------------------------------
# Load raw data
# --------------------------------------------------------------
files = glob(full_path_pattern)

# Take every 7th file for the test dataset
test_files = [f for i, f in enumerate(files) if i % 7 == 1]
