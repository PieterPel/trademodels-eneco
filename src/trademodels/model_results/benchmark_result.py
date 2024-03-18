from typing import List
from .model_result import ModelResult
from ..dataclasses import (
    Order,
    ProcessedData,
    TradingSignal,
    TradingSignals,
    OrderType,
    OrderSide,
)
import pandas as pd


class BenchmarkExecutionResult(ModelResult):
    def __init__(self, interval_length: int):
        self.interval_length = interval_length

    def get_output(
        self, _: ProcessedData, signals: TradingSignals
    ) -> List[Order]:
        orders = [
            Order(
                OrderType.MARKET_ORDER,
                signal.side,
                signal.size,
                None,
                signal.time,
            )
            for signal in signals
        ]

        return orders

    def summary(self):
        pass


class BenchmarkTradingSignalResult(ModelResult):
    def __init__(self, interval_length: int):
        self.interval_length = interval_length

    def get_output(
        self, data: ProcessedData, order_size: int
    ) -> TradingSignals:
        timestamps = [
            interval[1]
            for interval in data.get_intervals_with_full_bid_and_ask(
                self.interval_length
            )
        ]
        signals = TradingSignals.always_buy(order_size, timestamps)
        return signals

    def summary(self):
        pass


def get_benchmark_trades(
    buy_trades: list[Order],
    sell_trades: list[Order],
    signals: TradingSignals,
    interval_length: int,
):
    benchmark_trades = []

    buy_dict = {
        order.time: len(buy_trades) - index - 1
        for index, order in enumerate(buy_trades[::-1])
    }
    sell_dict = {
        order.time: len(sell_trades) - index - 1
        for index, order in enumerate(sell_trades[::-1])
    }

    for signal in signals:

        if str(signal.side) == str(OrderSide.BID):
            collected_trades = collect_trades(
                signal, buy_trades, buy_dict, interval_length
            )

        elif str(signal.side) == str(OrderSide.ASK):
            collected_trades = collect_trades(
                signal, sell_trades, sell_dict, interval_length
            )

        benchmark_trades.extend(collected_trades)

    return benchmark_trades


def collect_trades(
    signal: TradingSignal,
    trades: list[Order],
    dict_: dict,
    interval_length: int,
):
    collected_trades = []

    time = signal.time
    end_time = signal.time + pd.Timedelta(minutes=interval_length)

    while time <= end_time:

        if time in dict_.keys():
            index = dict_[time]

            while time <= end_time and index < len(trades):
                order = trades[index]

                if order.time < end_time and not (
                    order.time == signal.time
                    and str(order.type) == str(OrderType.MARKET_ORDER)
                ):
                    collected_trades.append(order)
                elif (
                    order.time == end_time
                    and order.type == OrderType.MARKET_ORDER
                ):
                    collected_trades.append(order)
                    break

                time = order.time
                index += 1

        time += pd.Timedelta(minutes=interval_length)

    return collected_trades
