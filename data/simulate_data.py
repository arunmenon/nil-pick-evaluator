# data/simulate_data.py
"""
Provides functionality to generate synthetic inventory and demand data for evaluation.
This simulates daily inventory levels, demand, sales, nil-picks, etc.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_inventory_data(num_stores=2, num_skus=2, days=60, 
                                      reorder_point=30, reorder_quantity=100, lead_time_days=5, seed=42):
    """
    Generate synthetic daily inventory, demand, sales data for multiple stores and SKUs.

    Parameters
    ----------
    num_stores : int
        Number of stores.
    num_skus : int
        Number of SKUs.
    days : int
        Number of days of data to simulate.
    reorder_point : int
        Reorder point for inventory policy.
    reorder_quantity : int
        Reorder quantity when below reorder point.
    lead_time_days : int
        How many days before an order arrives.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: date, store, sku, promotion_flag, demand, sales, nil_picks,
        starting_inventory, ending_inventory, item_id
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2022,1,1)
    stores = [f"store_{s:03d}" for s in range(num_stores)]
    skus = [f"sku_{k:03d}" for k in range(num_skus)]

    store_factor = rng.normal(loc=1.0, scale=0.1, size=num_stores)
    sku_popularity = rng.normal(loc=50, scale=20, size=num_skus)

    data_records = []
    inventory_levels = {(store,sku):100 for store in stores for sku in skus}
    pending_orders = {(store,sku):[] for store in stores for sku in skus}

    for d in range(days):
        current_date = start_date + timedelta(days=d)
        day_of_year = current_date.timetuple().tm_yday
        seasonality = np.sin(2*np.pi*day_of_year/365)*20

        for i, store in enumerate(stores):
            for j, sku in enumerate(skus):
                # Receive orders if any arrive today
                arrivals_today = [q for arr_day, q in pending_orders[(store, sku)] if arr_day == d]
                if arrivals_today:
                    inv = inventory_levels[(store,sku)]
                    inventory_levels[(store,sku)] = inv + sum(arrivals_today)
                    pending_orders[(store,sku)] = [(arr_day, q) for arr_day, q in pending_orders[(store,sku)] if arr_day != d]

                # Compute demand
                base_demand = max(sku_popularity[j] * store_factor[i] + seasonality, 0)
                noise = rng.normal(scale=5)
                promotion_flag = (rng.random() < 0.05)
                demand = base_demand * (1.5 if promotion_flag else 1.0) + noise
                demand = max(int(round(demand)), 0)

                current_inventory = inventory_levels[(store,sku)]
                if demand > current_inventory:
                    sales = current_inventory
                    nil_picks = demand - current_inventory
                    ending_inv = 0
                else:
                    sales = demand
                    nil_picks = 0
                    ending_inv = current_inventory - sales

                inventory_levels[(store,sku)] = ending_inv

                # Check reorder
                if ending_inv < reorder_point:
                    arrival_day = d + lead_time_days
                    pending_orders[(store,sku)].append((arrival_day, reorder_quantity))

                data_records.append({
                    "date": current_date,
                    "store": store,
                    "sku": sku,
                    "promotion_flag": int(promotion_flag),
                    "demand": demand,
                    "sales": sales,
                    "nil_picks": nil_picks,
                    "starting_inventory": current_inventory,
                    "ending_inventory": ending_inv
                })

    df = pd.DataFrame(data_records)
    df["item_id"] = df["store"] + "_" + df["sku"]
    df.set_index("date", inplace=True)
    return df

