import pandas as pd


def make_features(df):
    """Leave those features in the dataframe,
    that are important to the model

    Args:
        df (pd.DataFrame): The processed dataframe

    Returns:
        pd.DataFrame: Part of the input dataframe with important features
    """

    selected_features = [
        "acc_y_temp_mean_ws_5",
        "gyr_r_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.0_Hz_ws_14",
        "duration",
    ]

    df_selected = df[selected_features]

    return df_selected
