from .model import ExecutionModel, TradingSignalModel
from ..model_results.benchmark_result import (
    BenchmarkExecutionResult,
    BenchmarkTradingSignalResult,
)


class BenchmarkExecutionModel(ExecutionModel):
    def train(self, interval_length: int, _: int) -> BenchmarkExecutionResult:
        return BenchmarkExecutionResult(interval_length)


class BenchmarkTradingSignalModel(TradingSignalModel):
    def train(self, interval_length: int) -> BenchmarkTradingSignalResult:
        return BenchmarkTradingSignalResult(interval_length)
