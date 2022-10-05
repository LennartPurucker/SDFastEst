import pandas as pd


# --- Preprocessing Steps
def discretize(data, force_string_for_small=False, str_threshold=5):
    """
    Discretize input data - currently using pandas qcut
    - Return as string to make it compatible with non-pandas software
    """
    # If the data has only a few numeric values, force to be strings instead of binning
    if force_string_for_small and (data.nunique() <= str_threshold):
        return data.astype(str)

    return pd.qcut(data, q=4, duplicates="drop").astype(str)


def handle_missing_values(df, missing_value_indicator, axis=0):
    # Drop instances with missing values
    for col in df.columns:
        comparer_col = df[col].astype(str)
        drop_list = df.loc[comparer_col == missing_value_indicator].index

        if axis == 0:
            # Delete these row indexes from dataFrame
            df.drop(drop_list, inplace=True, axis=axis)
        elif not drop_list.empty:
            # Drop column with missing values
            df.drop(columns=col, inplace=True)
