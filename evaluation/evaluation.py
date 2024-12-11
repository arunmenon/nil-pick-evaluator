# evaluation/evaluation.py
"""
Contains functions to evaluate forecasts for the nil-pick scenario:
- Calculate nil-picks and service level (fill rate) given actual demands and predicted inventories.
"""

import numpy as np

def calculate_nil_picks_and_service_level(actual_demand, predicted_inventory):
    """
    Calculate number of nil-picks and service level given actual demand and predicted inventory.

    Parameters
    ----------
    actual_demand : np.array
        Array of actual demands for the forecast period.
    predicted_inventory : np.array
        Array of predicted inventory levels for the forecast period.

    Returns
    -------
    nil_picks : int
        Count of how many instances where predicted_inventory < actual_demand.
    service_level : float
        (Fulfilled demand / Total demand) * 100.
    """
    fulfilled = np.minimum(predicted_inventory, actual_demand)
    nil_picks = np.sum(predicted_inventory < actual_demand)
    total_demand = actual_demand.sum()
    service_level = (fulfilled.sum() / total_demand * 100) if total_demand > 0 else 100.0
    return nil_picks, service_level

