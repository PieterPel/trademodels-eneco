from abc import ABC, abstractmethod
from ..dataclasses import ProcessedData
from ..model_results.model_result import ModelResult


class Model(ABC):
    def __init__(self, data: ProcessedData):
        self.data = data

    @abstractmethod
    def train(self, interval_lenght: int) -> ModelResult:
        ...


class ExecutionModel(Model):
    """
    Model for optimal execution of trading signals
    """


class TradingSignalModel(Model):
    """
    Model to get trading signals
    """
