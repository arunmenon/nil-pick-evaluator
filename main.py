# main.py
"""
Main orchestration script:
1. Generate synthetic inventory data.
2. Prepare dataset for forecasting (next-day predictions of inventory).
3. Run inference using Lag-Llama.
4. Evaluate nil-pick metrics.
"""

import numpy as np
from data.simulate_data import generate_synthetic_inventory_data
from utils.dataset_builder import build_pandas_dataset
from models.lag_llama_model import LagLlamaModel
from evaluation.evaluation import calculate_nil_picks_and_service_level

def main():
    # Step 1: Generate synthetic data
    df = generate_synthetic_inventory_data(num_stores=2, num_skus=2, days=60)
    target_col = "ending_inventory"

    # Split data into train and test periods
    sorted_dates = df.index.unique().sort_values()
    train_cutoff = int(len(sorted_dates)*0.8)
    train_dates = sorted_dates[:train_cutoff]
    test_dates = sorted_dates[train_cutoff:]

    # For simplicity, we just use the full dataset.
    # In a realistic scenario, you'd create a backtest scenario.
    dataset = build_pandas_dataset(df.reset_index(), target_col=target_col, timestamp_col="date", item_id_col="item_id")

    # Step 2: Initialize model
    # prediction_length=1 for next-day forecasts
    model = LagLlamaModel(
        ckpt_path="lag-llama.ckpt", 
        prediction_length=1,
        context_length=32,
        use_rope_scaling=False,
        num_samples=100,
        device="cpu"
    )

    # Step 3: Get predictions
    forecasts, tss = model.predict(dataset)

    # Convert probabilistic forecasts to point forecasts (mean)
    point_forecasts = [fc.samples.mean(axis=0)[0] for fc in forecasts]

    # Align actual demands with forecast timestamps:
    # Each forecast corresponds to a certain item_id and a future timestamp.
    # Let's extract actual demands from df for the forecast start:
    actual_demands = []
    predicted_inventories = point_forecasts

    for fc in forecasts:
        forecast_start = fc.start_date.to_timestamp()
        item_id = fc.item_id
        actual_row = df.loc[(df["item_id"] == item_id) & (df.index == forecast_start)]
        if len(actual_row) == 0:
            # If no actual row found (end of data), assume zero demand or skip
            actual_demands.append(0)
        else:
            actual_demands.append(actual_row["demand"].values[0])

    actual_demands = np.array(actual_demands)
    predicted_inventories = np.array(predicted_inventories)

    # Step 4: Evaluate metrics
    nil_picks, service_level = calculate_nil_picks_and_service_level(actual_demands, predicted_inventories)
    print("Evaluation Results:")
    print("Nil-Picks:", nil_picks)
    print("Service Level (%):", service_level)

if __name__ == "__main__":
    main()

