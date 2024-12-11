# models/model_interface.py
"""
Defines a common interface for time-series forecasting models. 
Any model used for evaluating nil-picks should implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List

class ModelInterface(ABC):
    @abstractmethod
    def predict(self, dataset) -> (List, List):
        """
        Given a dataset (in a GluonTS-compatible format), produce forecasts and their corresponding 
        ground truth time series (ts).

        Parameters
        ----------
        dataset : gluonts.dataset.Dataset
            A dataset that the model will forecast.

        Returns
        -------
        forecasts : List
            A list of GluonTS forecast objects for each time series in the dataset.
        tss : List
            A list of time series arrays, each corresponding to a forecast in forecasts.
        """
        pass
