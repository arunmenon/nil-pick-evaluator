# models/lag_llama_model.py
"""
Implementation of the Lag-Llama model using the ModelInterface.

This class encapsulates the logic for running inference using Lag-Llama, 
making it easy to switch to another model later.
"""

import torch
from typing import List
from models.model_interface import ModelInterface

# Ensure you have Lag-Llama's estimator and dependencies installed
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions

class LagLlamaModel(ModelInterface):
    def __init__(self, 
                 ckpt_path: str,
                 prediction_length: int,
                 context_length: int = 32,
                 use_rope_scaling: bool = False,
                 num_samples: int = 100,
                 device: str = "cpu"):
        """
        Initialize the Lag-Llama model wrapper.

        Parameters
        ----------
        ckpt_path : str
            Path to the Lag-Llama checkpoint file (e.g., lag-llama.ckpt).
        prediction_length : int
            Forecast horizon (e.g., 1 for next-day prediction).
        context_length : int, optional
            Context length for the model. Lag-Llama default is 32.
        use_rope_scaling : bool, optional
            Whether to use rope scaling for handling longer contexts.
        num_samples : int, optional
            Number of samples for probabilistic forecasts.
        device : str, optional
            The device to run inference on (e.g., 'cuda:0' or 'cpu').
        """
        self.ckpt_path = ckpt_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.use_rope_scaling = use_rope_scaling
        self.num_samples = num_samples
        self.device = torch.device(device)

        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.estimator_args = self.ckpt["hyper_parameters"]["model_kwargs"]

    def predict(self, dataset) -> (List, List):
        """
        Generate forecasts and return (forecasts, tss).
        """
        forecasts, tss = self._run_inference(dataset)
        return forecasts, tss

    def _run_inference(self, dataset):
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (self.context_length + self.prediction_length) / self.estimator_args["context_length"]),
        }

        estimator = LagLlamaEstimator(
            ckpt_path=self.ckpt_path,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            input_size=self.estimator_args["input_size"],
            n_layer=self.estimator_args["n_layer"],
            n_embd_per_head=self.estimator_args["n_embd_per_head"],
            n_head=self.estimator_args["n_head"],
            scaling=self.estimator_args["scaling"],
            time_feat=self.estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments if self.use_rope_scaling else None,
            batch_size=1,
            num_parallel_samples=self.num_samples,
            device=self.device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=self.num_samples
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        return forecasts, tss
