from __future__ import annotations
import pandas as pd
from enum import Enum, auto
from typing import NamedTuple
from collections import UserList
from .utils import get_ml_encoder
import copy
import numpy as np

from sklearn.preprocessing import LabelEncoder


class RawData(pd.DataFrame):
    """
    Represents raw data for trading models.

    This class extends the functionality of a pandas DataFrame and provides additional methods for data processing.

    Attributes:
        MIN_HOUR (int): The minimum hour of the trading hours.
        MAX_HOUR (int): The maximum hour of the trading hours.

    Methods:
        change_to_datetime(): Converts the 'from_timestamp' and 'to_timestamp' columns to datetime format.
        get_filled(): Returns a new instance of `RawData` with missing values filled using forward fill method.
        remove_hours(min_hour: int, max_hour: int, inplace: bool = False) -> "RawData": Removes rows from the DataFrame based on the hour of the 'to_timestamp' column.
        process(min_hour: int = None, max_hour: int = None) -> "ProcessedData": Process the data by removing hours within the specified range.
    """

    MIN_HOUR = 8
    MAX_HOUR = 18
    # TODO: probably shouldn't hardcode these like this

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # TODO: Raise error if incorrect columns

    def change_to_datetime(self) -> None:
        """
        Converts the 'from_timestamp' and 'to_timestamp' columns to datetime format.
        """
        self["from_timestamp"] = pd.to_datetime(self["from_timestamp"])
        self["to_timestamp"] = pd.to_datetime(self["to_timestamp"])

    def get_filled(self) -> RawData:
        """
        Returns a new instance of `RawData` with missing values filled using forward fill method.

        Returns:
            RawData: A new instance of `RawData` with missing values filled.
        """
        df = copy.copy(self)
        # Set 'from_timestamp' as the index
        df.set_index("from_timestamp", inplace=True)

        # Create a mask for the newly created rows during forward fill
        mask_original = df["volume"].isna().copy()

        # Group by day and perform forward fill within each group
        df = RawData(
            df.groupby(df.index.date).apply(
                lambda group: group.resample("T").ffill()
            )
        )

        # Set 'volume' to zero in the newly created rows
        df["volume"] = df["volume"].where(~mask_original, 0)

        # Reset the index of the final result
        df.reset_index(inplace=True)

        # Make sure the end timestamps are correct
        df["to_timestamp"] = df["from_timestamp"] + pd.Timedelta(minutes=1)

        # Drop weird shadow column
        df.drop("level_0", axis=1, inplace=True)

        return df

    def remove_hours(
        self, min_hour: int, max_hour: int, inplace: bool = False
    ) -> RawData:
        """
        Remove rows from the DataFrame based on the hour of the 'to_timestamp' column.

        Args:
            min_hour (int): The minimum hour (inclusive) to keep.
            max_hour (int): The maximum hour (exclusive) to keep.
            inplace (bool, optional): If True, modify the DataFrame in place. If False, return a new DataFrame. Defaults to False.

        Returns:
            RawData: The modified DataFrame.

        """
        self.drop(
            self[
                (self["to_timestamp"].dt.hour < min_hour)
                | (self["to_timestamp"].dt.hour >= max_hour)
            ].index,
            inplace=inplace,
        )
        return self

    # Deprecated
    def remove_intervals_with_bid_or_ask_na(
        self, interval_length: int, inplace: bool = False
    ) -> None:
        df = copy.copy(self)

        # Set 'from_timestamp' as the index
        df.set_index("from_timestamp", inplace=inplace)

        # Create a list to store selected indices
        selected_indices = []

        # Define the time range for iteration
        start_time = df.index[0]
        end_time = df.index[-1]

        # Specify the interval length (5 minutes)
        interval_length = pd.Timedelta(minutes=interval_length - 1)
        total_length = 0

        # Iterate through the data in five-minute intervals
        while start_time <= end_time:
            end_interval = start_time + interval_length

            # Extract the five-minute interval
            interval_data = df.loc[start_time:end_interval]

            # Get the next start time
            start_time = start_time + interval_length + pd.Timedelta(minutes=1)

            if len(interval_data) < 5:
                continue

            # Check if both 'best_bid' and 'best_ask' are available for
            # every minute in the interval
            if not (
                interval_data["best_bid"].isna().any()
                or interval_data["best_ask"].isna().any()
            ):
                selected_indices.extend(interval_data.index)
                total_length += len(interval_data)

            if start_time.hour >= 18:
                # Set start_time to 8:00 the next day
                start_time = start_time + pd.Timedelta(days=1)
                start_time = start_time.replace(hour=8, minute=0)

        # Filter the original DataFrame based on selected indices
        df = df.loc[selected_indices]
        df.reset_index(inplace=True)

        return df

    def process(
        self,
        min_hour: int = None,
        max_hour: int = None,
    ) -> ProcessedData:
        """
        Process the data by removing hours within the specified range.

        Args:
            min_hour (int, optional): The minimum hour to include in the processed data. If not provided, the default minimum hour will be used.
            max_hour (int, optional): The maximum hour to include in the processed data. If not provided, the default maximum hour will be used.

        Returns:
            ProcessedData: The processed data with the specified hours removed.
        """
        if min_hour is None:
            min_hour = self.MIN_HOUR
        if max_hour is None:
            max_hour = self.MAX_HOUR

        self.change_to_datetime()
        filled = self.get_filled()
        filled.remove_hours(min_hour, max_hour, inplace=True)

        return ProcessedData(filled)


class ProcessedData(pd.DataFrame):
    """
    A subclass of pd.DataFrame representing processed data for trading models.

    This class provides additional methods for data processing and manipulation
    specific to trading models, such as interpolation, selecting intervals with
    complete bid and ask data, and preparing data for machine learning and
    reinforcement learning.

    Attributes:
        intervals_cache (list): A cache for selected intervals with complete bid
            and ask data.
        interpolated_cache (pd.DataFrame): A cache for the interpolated DataFrame.
        ready_for_ml_cache (pd.DataFrame): A cache for the DataFrame prepared for
            machine learning.
        ready_for_rl_cache (pd.DataFrame): A cache for the DataFrame prepared for
            reinforcement learning.

    Methods:
        get_interpolated_df: Returns the interpolated DataFrame.
        get_intervals_with_full_bid_and_ask: Retrieves intervals with both 'best_bid'
            and 'best_ask' available for every minute.
        get_df_for_ml: Returns a DataFrame suitable for machine learning with lagged
            features.
        get_df_for_rl: Returns a DataFrame suitable for reinforcement learning with
            lagged features.
        get_reversed_df: Returns a deep copy of the DataFrame with the values in the
            non-timestamp columns reversed.
        reset_cache: Resets the cache for intervals, interpolated data, and readiness
            for machine learning and reinforcement learning.
        from_raw_data: Create a ProcessedData object from raw data.
    """

    intervals_cache = None
    interpolated_cache = None
    ready_for_ml_cache = None
    ready_for_rl_cache = None

    @property
    # this method is makes it so our methods return an instance
    # of MaturitiesDf, instead of a regular DataFrame
    def _constructor(self):
        return ProcessedData

    def get_interpolated_df(
        self, use_cache: bool = True, *args, **kwargs
    ) -> ProcessedData:
        """
        Returns the interpolated DataFrame.

        Args:
            use_cache (bool, optional): Flag indicating whether to use the cached interpolated DataFrame. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ProcessedData: The interpolated DataFrame.
        """
        if not use_cache or self.interpolated_cache is None:
            self.interpolated_cache = copy.copy(
                self.interpolate(method="pad", *args, **kwargs)
            )

        return self.interpolated_cache

    def get_intervals_with_full_bid_and_ask(
        self,
        interval_length: int,
        use_cache: bool = True,
        min_hour: int = 8,
        max_hour: int = 18,
    ) -> list:
        """
        Retrieves intervals with both 'best_bid' and 'best_ask' available for every minute.

        Args:
            interval_length (int): The length of the interval in minutes.
            use_cache (bool, optional): Whether to use the cached intervals if available. Defaults to True.
            min_hour (int, optional): The minimum hour of the day to consider. Defaults to 8.
            max_hour (int, optional): The maximum hour of the day to consider. Defaults to 18.

        Returns:
            list: A list of selected intervals with both 'best_bid' and 'best_ask' available for every minute.
        """
        if use_cache and self.intervals_cache is not None:
            return self.intervals_cache

        print(
            "Obtaining the intervals that we can use, this will take some time"
        )

        # Set 'from_timestamp' as the index
        self.set_index("from_timestamp", inplace=True)

        # Create a list to store selected indices
        selected_intervals = []

        # Define the time range for iteration
        start_time = self.index[0]
        start_time = start_time.replace(
            minute=(start_time.minute // interval_length) * interval_length
        )
        end_time = self.index[-1]

        # Specify the interval length (5 minutes)
        interval_length = pd.Timedelta(minutes=interval_length)

        # Iterate through the data in five-minute intervals
        while start_time <= end_time:
            # TODO: get rid of these hardocdes
            if start_time.hour >= max_hour:
                # Set start_time to 8:00 the next day
                start_time = start_time + pd.Timedelta(days=1)
                start_time = start_time.replace(hour=min_hour, minute=0)

            end_interval = start_time + interval_length

            # Extract the five-minute interval
            interval_data = self.loc[
                (start_time - pd.Timedelta(minutes=1)) : end_interval
            ]

            # Get the next start time
            start_time = start_time + interval_length

            if len(interval_data) < 6:
                continue

            # Check if both 'best_bid' and 'best_ask' are available
            # for every minute in the interval
            if not (
                interval_data["best_bid"].isna().any()
                or interval_data["best_ask"].isna().any()
            ):
                selected_intervals.append(interval_data.index)

        self.reset_index(inplace=True)

        self.intervals_cache = selected_intervals

        print("Done")

        return selected_intervals

    def get_df_for_ml(
        self,
        interval_length: int,
        num_lags: int,
        encoder: LabelEncoder = None,
        fit_encoder: bool = True,
        use_cache: bool = True,
        holdable: bool = False,
        keep_no_price_change: bool = True,
    ) -> "ProcessedData":
        # TODO: add variable for time of the day

        if use_cache and self.ready_for_ml_cache is not None:
            return self.ready_for_ml_cache

        if encoder is None:
            encoder = LabelEncoder()

        interpolated_data = self.get_interpolated_df()

        interpolated_data["minutes_in_day"] = (
            interpolated_data["to_timestamp"].dt.hour * 60
            + interpolated_data["to_timestamp"].dt.minute
        )

        legal_intervals = self.get_intervals_with_full_bid_and_ask(
            interval_length
        )

        legal_interval_starts = [
            interval[0]
            for interval in legal_intervals
            if interval[0].hour >= 8 and interval[0].minute > 0
        ]

        data_time_index = self.set_index("from_timestamp")

        optimal_actions = []
        no_change_interval_starts = []

        for interval_start in legal_interval_starts:
            try:
                current_best_bid = data_time_index.loc[
                    interval_start - pd.Timedelta(minutes=1)
                ]["best_bid"]
                current_best_ask = data_time_index.loc[
                    interval_start - pd.Timedelta(minutes=1)
                ]["best_ask"]

                future_best_bid = data_time_index.loc[
                    interval_start
                    - pd.Timedelta(minutes=1)
                    + pd.Timedelta(minutes=interval_length)
                ]["best_bid"]
                future_best_ask = data_time_index.loc[
                    interval_start
                    - pd.Timedelta(minutes=1)
                    + pd.Timedelta(minutes=interval_length)
                ]["best_ask"]
            except KeyError:
                no_change_interval_starts.append(interval_start)
                continue

            # Look at price difference in 5 minutes
            current_price = (current_best_ask + current_best_bid) / 2
            future_price = (future_best_ask + future_best_bid) / 2
            current_spread = current_best_ask - current_best_bid

            # If increase larger than spread -> buy
            if future_price - current_price > current_spread * holdable:
                action = actions.BUY
                optimal_actions.append(action)

            # If decrease larger than spread -> sell
            elif future_price - current_price < -current_spread * holdable:
                action = actions.SELL
                optimal_actions.append(action)

            # Else: hold
            else:
                action = actions.HOLD

                if holdable or keep_no_price_change:
                    optimal_actions.append(action)
                else:
                    no_change_interval_starts.append(interval_start)

        interpolated_data_with_lags = interpolated_data.set_index(
            "from_timestamp"
        ).copy()

        columns_to_lag = ["open", "high", "low", "close", "volume"]

        add_lags(interpolated_data_with_lags, columns_to_lag, num_lags)

        legal_interval_starts = [
            interval_start
            for interval_start in legal_interval_starts
            if interval_start not in no_change_interval_starts
        ]

        trimmed_df = interpolated_data_with_lags.loc[legal_interval_starts]

        trimmed_df["optimal_action"] = optimal_actions

        # Make all prices relative to the current best bid
        no_substract_columns = [
            "from_timestamp",
            "to_timestamp",
            "optimal_action",
            "minutes_in_day",
        ]
        columns_to_substract = [
            col
            for col in trimmed_df.columns
            if (col not in no_substract_columns and "volume" not in col)
        ]

        trimmed_df[columns_to_substract] = trimmed_df[
            columns_to_substract
        ].sub(trimmed_df["best_bid"], axis=0)

        # Encode the 'optimal_action' column
        trimmed_df["enum_column"] = trimmed_df["optimal_action"].apply(
            lambda x: x.name
        )  # Convert enum to its name

        # encoder = get_ml_encoder()

        if fit_encoder:  # NOTE check if this works otherwise remove
            trimmed_df["optimal_action_encoded"] = encoder.fit_transform(
                trimmed_df["enum_column"]
            )
        else:
            trimmed_df["optimal_action_encoded"] = encoder.transform(
                (trimmed_df["enum_column"])
            )

        self.ready_for_ml_cache = trimmed_df

        return trimmed_df

    def get_df_for_rl(self, num_lags: int, use_cache: bool = True):
        """
        Returns a DataFrame suitable for reinforcement learning (RL) with lagged features.

        Args:
            num_lags (int): The number of lagged features to include in the DataFrame.
            use_cache (bool, optional): Whether to use the cached DataFrame if available. Defaults to True.

        Returns:
            pandas.DataFrame: The DataFrame with lagged features suitable for RL.
        """
        if use_cache and self.ready_for_rl_cache is not None:
            return self.ready_for_rl_cache

        interpolated_data = self.get_interpolated_df()

        interpolated_data["minutes_in_day"] = (
            interpolated_data["to_timestamp"].dt.hour * 60
            + interpolated_data["to_timestamp"].dt.minute
        )

        interpolated_data_with_lags = interpolated_data.set_index(
            "from_timestamp"
        ).drop("to_timestamp", axis=1)

        columns_to_lag = ["open", "high", "low", "close", "volume"]

        add_lags(interpolated_data_with_lags, columns_to_lag, num_lags)

        self.ready_for_rl_cache = interpolated_data_with_lags

        return interpolated_data_with_lags

    def get_reversed_df(self):
        """
        Returns a deep copy of the DataFrame with the values in the non-timestamp columns reversed.

        Returns:
            DataFrame: A deep copy of the DataFrame with the values in the non-timestamp columns reversed.
        """
        df = copy.deepcopy(self)

        columns_to_reverse = df.columns.difference(
            ["from_timestamp", "to_timestamp"]
        )
        df[columns_to_reverse] = df[columns_to_reverse].values[::-1]

        # Swap "open" and "close" columns
        df["open"], df["close"] = df["close"], df["open"]

        # Move "best_bid" and "best_ask" values up one row
        df["best_bid"] = df["best_bid"].shift(-1)
        df["best_ask"] = df["best_ask"].shift(-1)

        # Drop the last row
        df = df.iloc[:-1]

        if "index" in df.columns:
            df.drop("index", axis=1, inplace=True)

        df.reset_cache()

        return df

    def intervals_with_constant_bid_ask(
        self, intervals: list, interval_length=5
    ) -> list:

        df = self.get_df_for_rl(interval_length)
        change = False

        constant_intervals = []

        for interval in intervals:
            first_minute = interval[0]

            first_bid = df.loc[first_minute]["best_bid"]
            first_ask = df.loc[first_minute]["best_ask"]

            for minute in interval[1:]:

                if df.loc[minute]["best_bid"] != first_bid:
                    change = True
                    break

                if df.loc[minute]["best_ask"] != first_ask:
                    change = True
                    break

            if not change:
                constant_intervals.append(interval)

            change = False

        return constant_intervals

    def intervals_with_same_bid_ask_begin_and_end(
        self, intervals: list, interval_length: int = 5
    ) -> list:
        df = self.get_df_for_rl(interval_length)

        same = []

        for interval in intervals:

            interval_begin = interval[0]
            interval_end = interval[-1]

            if (
                abs(
                    df.loc[interval_begin]["best_bid"]
                    - df.loc[interval_end]["best_bid"]
                )
                < 0.01
                and abs(
                    df.loc[interval_begin]["best_ask"]
                    - df.loc[interval_end]["best_ask"]
                )
                < 0.01
            ):
                same.append(interval)

        return same

    def reset_cache(self):
        """
        Resets the cache for intervals, interpolated data, and readiness for machine learning and reinforcement learning.
        """
        self.intervals_cache = None
        self.interpolated_cache = None
        self.ready_for_ml_cache = None
        self.ready_for_rl_cache = None

    @classmethod
    def from_raw_data(cls, *args, **kwargs) -> ProcessedData:
        """
        Create a ProcessedData object from raw data.

        Args:
            *args: Positional arguments to be passed to the RawData constructor.
            **kwargs: Keyword arguments to be passed to the RawData constructor.

        Returns:
            ProcessedData: The processed data object.

        """
        raw_data = RawData(*args, **kwargs)
        return raw_data.process()

    def plot(self):
        pass


def add_lags(df, columns, n_lags):
    """
    Adds lagged columns to a DataFrame for specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame to add lagged columns to.
        columns (list): List of column names to add lagged columns for.
        n_lags (int): Number of lagged columns to add.

    Returns:
        None
    """
    for column in columns:
        for lag in range(1, n_lags + 1):
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)


class actions(Enum):
    """
    Enum representing the actions the model can take in an interval
    """

    BUY = auto()
    SELL = auto()
    HOLD = auto()


class OrderType(Enum):
    """
    Enum representing the type of an order
    """

    LIMIT_ORDER = auto()
    MARKET_ORDER = auto()
    CONDITIONAL_LIMIT_ORDER = (
        auto()
    )  # would require an extra field in Order for the condition probably


class OrderSide(Enum):
    """
    Enum representing the side of an order.
    """

    BID = auto()
    ASK = auto()


class TradingSignal(NamedTuple):
    """
    Represents a trading signal.

    Attributes:
        side (OrderSide): The side of the order (buy or sell).
        size (int): The size of the order.
        time (pd.Timestamp): The timestamp of the signal.
    """

    side: OrderSide
    size: int
    time: pd.Timestamp


class Order(NamedTuple):
    """
    Represents an order in the trading system.

    Attributes:
        type (OrderType): The type of the order (e.g., limit, market).
        side (OrderSide): The side of the order (e.g., buy, sell).
        volume (int): The volume of the order.
        price (float): The price of the order. None if it is a market order.
        time (pd.Timestamp): The timestamp of the order.
    """

    type: OrderType
    side: OrderSide
    volume: int
    price: float  # None if it is a market order
    time: pd.Timestamp


class OrderRL(NamedTuple):
    """
    Represents an order in a reinforcement learning trading model.

    Attributes:
        volume (int): The volume of the order.
        price (int): The price of the order.
    """

    volume: int
    price: int


class TradingSignals(UserList):
    """
    Represents a collection of trading signals.

    Inherits from UserList, providing a list-like interface for managing trading signals.

    Usage:
    signals = TradingSignals(always_buy(10, [time1, time2, time3]))
    """

    @classmethod
    def always_buy(cls, order_size, times):
        signals = [
            TradingSignal(OrderSide.BID, order_size, time) for time in times
        ]

        return cls(signals)

    @classmethod
    def always_sell(cls, order_size, times):
        signals = [
            TradingSignal(OrderSide.ASK, order_size, time) for time in times
        ]

        return cls(signals)


class PctOff(NamedTuple):
    """
    Represents a percentage off at a specific interval.

    Attributes:
        interval_end (pd.Timestamp): The end timestamp of the interval.
        pct_off (float): The percentage off.
    """

    interval_end: pd.Timestamp
    pct_off: float


class RLResults(NamedTuple):
    """
    Represents the results of a reinforcement learning trading model.

    Attributes:
        inserted_orders (list): A list of inserted orders.
        executed_limit_orders (list): A list of executed limit orders.
        signal_rewards (list): A list of signal rewards.
        execution_rewards (list): A list of execution rewards.
        execution_rewards_bm (list): A list of execution rewards benchmarked.
        benchmark_orders (list): A list of benchmark orders.
        times (list): A list of times.
    """

    inserted_orders: list
    executed_limit_orders: list
    signal_rewards: list
    execution_rewards: list
    execution_rewards_bm: list
    benchmark_orders: list
    times: list
