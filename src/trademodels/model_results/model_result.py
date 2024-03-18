from abc import ABC, abstractmethod


class ModelResult(ABC):
    interval_length: int

    @abstractmethod
    def get_output(self):
        ...

    @abstractmethod
    def summary(self):
        ...
