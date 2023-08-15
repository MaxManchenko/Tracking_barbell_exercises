import json
from glob import glob

from make_preprocessed_data import DataPreProcessor
from make_processed_data import DataProcessor
from make_features import make_features

config_path = "config.json"


# --------------------------------------------------------------
# Load config. settings, Build the path to the files and Load data
# --------------------------------------------------------------
with open(config_path, "r") as config_file:
    config = json.load(config_file)

files_path = config["files_path"]
file_pattern = config["file_pattern"]
full_path_pattern = files_path + "*" + file_pattern
path_to_save_processed_data = config["data_path_fully_processed"]

files = glob(full_path_pattern)

# --------------------------------------------------------------
# Preprocess data
# --------------------------------------------------------------
# Call DataPreProcessor
data_preprocessor = DataPreProcessor()

# Read data
acc_df, gyr_df = data_preprocessor.read_data_from_files(files)

# Merge data
data_merged = data_preprocessor.merge_dataframes(acc_df, gyr_df)

# Resample data
data_resampled = data_preprocessor.resample_data(data_merged)

# --------------------------------------------------------------
# Process data
# --------------------------------------------------------------
# Call DataProcessor
data_processor = DataProcessor()

# Mark outliers
data_w_marked_outliers = data_processor.remove_outliers_by_IQR(data_resampled)

# Impute missing values
data_w_missing_data_filled_in = data_processor.impute_missing_values(
    data_w_marked_outliers
)

# Calculate set duration
data_w_set_duration = data_processor.calculate_set_duration(
    data_w_missing_data_filled_in
)

# Apply lowpass filter
data_lowpass = data_processor.lowpass_filter(data_w_set_duration)

# Add square attributes
data_w_square_attr = data_processor.square_attributes(data_lowpass)

# Calculate rolling average
data_temporal = data_processor.rolling_averege(data_w_square_attr)

# Make Fourier transformation
data_freq = data_processor.fourier_transform(data_temporal)

# --------------------------------------------------------------
# Build the final dataset with selected features
# --------------------------------------------------------------
df_selected = make_features(data_freq)

# --------------------------------------------------------------
# Save the processed data
# --------------------------------------------------------------
df_selected.to_pickle(path_to_save_processed_data)
