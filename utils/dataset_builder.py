# utils/dataset_builder.py
"""
Utility functions to prepare a PandasDataset from a DataFrame for GluonTS forecasting.

We assume the DataFrame has 'date' as index, 'item_id', and 'ending_inventory' (or another target).
"""

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

def build_pandas_dataset(df: pd.DataFrame, target_col: str, timestamp_col: str = "date", item_id_col: str = "item_id"):
    """
    Build a PandasDataset from a long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that includes timestamps, target values, and item_ids.
    target_col : str
        Name of the column that contains the target variable to forecast.
    timestamp_col : str, optional
        Name of the timestamp column. If df is indexed by date, ensure it's reset and provided here.
    item_id_col : str, optional
        Name of the column that identifies different time series (e.g. SKU-store pairs).

    Returns
    -------
    dataset : PandasDataset
        A GluonTS-compatible dataset ready for forecasting.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_col} not found in DataFrame.")

    if item_id_col in df.columns:
        dataset = PandasDataset.from_long_dataframe(df.reset_index(), 
                                                    target=target_col, 
                                                    item_id=item_id_col, 
                                                    timestamp=timestamp_col)
    else:
        # Single time series case
        dataset = PandasDataset.from_long_dataframe(df.reset_index(), 
                                                    target=target_col, 
                                                    timestamp=timestamp_col)
    return dataset

