# Time-Series Forecasting Nil-Pick Evaluation Test Bed

This project provides a modular test bed for evaluating nil-pick (stockout) forecasting scenarios using large time-series 
foundation models like Lag-Llama. It can be easily extended to different models and datasets.

## Structure

- `models/`: Defines a `ModelInterface` and implements `LagLlamaModel`.  
- `data/`: Provides a `simulate_data.py` script to generate synthetic inventory and demand data.  
- `evaluation/`: Contains evaluation metrics for nil-picks and service level.  
- `utils/`: Helpers for converting DataFrame to `PandasDataset`.  
- `main.py`: Orchestrates the entire pipeline – data generation, forecasting, and evaluation.

## Steps

1. **Data Generation**: Synthetic inventory data is generated, including demand, inventory levels, and promotions.
2. **Dataset Preparation**: Converts the DataFrame into a `PandasDataset` compatible with GluonTS.
3. **Model Inference**: `LagLlamaModel` is used to produce forecasts.
4. **Evaluation**: Nil-picks and service level are computed from predicted vs. actual data.

## Extending the Solution

- **New Models**: Implement `ModelInterface` for another model (e.g., Lag-Laam), then swap it in `main.py`.
- **Real Data**: Replace the synthetic data generation with real historical data from your systems.
- **Additional Metrics**: Modify or add new metrics in `evaluation/` to suit your business KPIs.

## Requirements

- `gluonts`
- `torch`
- `numpy`
- `pandas`
- `matplotlib` (if needed for plotting)
- Lag-Llama dependencies

## Running

1. Ensure `lag-llama.ckpt` is available.
2. `python main.py`

The script will print evaluation results for nil-picks and service level.

