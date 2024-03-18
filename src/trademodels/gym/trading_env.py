from ..dataclasses import ProcessedData, OrderRL, Order, OrderSide, OrderType
from ..errors import AlwaysBuyAndAlwaysSellError

from typing import TypeVar
import itertools

import gymnasium as gym
import dill as pickle
from math import inf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

side_int_to_orderside = {0: OrderSide.BID, 1: OrderSide.ASK}


class TradingEnv(gym.Env):
    # type: dict[str, Any]
    metadata = {"render_modes": [None]}
    render_mode = None
    spec = None

    reward_range = (-inf, inf)

    def __init__(
        self,
        data: ProcessedData,
        num_lags: int,
        interval_length: int,
        max_volume: int,
        execution_weight: float = 1,
        signal_weight: float = 1,
        signal_multiplier: float = 1,
        holdable: bool = True,
        always_buy: bool = False,
        always_sell: bool = False,
        volume_penalizer: float = 0,
        cancel_after_one_minute: bool = False,
        intervals: list = None,
        stop_after_steps: int = -1,
        holding_execution_reward: float = 0,
    ):
        """
        Initialize the TradingEnvironment object.

        Args:
            data (ProcessedData): The processed data used for trading.
            num_lags (int): The number of lagged features to include in the trade data.
            interval_length (int): The length of each trading interval in minutes.
            max_volume (int): The maximum volume that can be traded in each interval.
            execution_weight (float, optional): The weight of execution-related factors in the trading model. Defaults to 1.
            signal_weight (float, optional): The weight of signal-related factors in the trading model. Defaults to 1.
            signal_multiplier (float, optional): The multiplier for determining the optimal side of an interval. Defaults to 1.
            holdable (bool, optional): Whether the assets can be held between intervals. Defaults to True.
            always_buy (bool, optional): Whether the model should always buy. Defaults to False.
            always_sell (bool, optional): Whether the model should always sell. Defaults to False.
            volume_penalizer (float, optional): The penalizer applied to the volume. Defaults to 1.

        Raises:
            AlwaysBuyAndAlwaysSellError: Raised when both always_buy and always_sell are set to True.

        """
        # TODO: remove non-legal intervals, turn into numpy array?
        self.trade_data = data.get_df_for_rl(num_lags)
        self.interval_length = interval_length
        self.max_volume = max_volume
        self.num_lags = num_lags
        self.original_data = data
        self.execution_weight = execution_weight
        self.signal_weight = signal_weight
        self.signal_multiplier = signal_multiplier
        self.holdable = holdable
        self.always_buy = always_buy
        self.always_sell = always_sell
        self.volume_penalizer = volume_penalizer
        self.cancel_after_one_minute = cancel_after_one_minute
        self.stop_after_steps = stop_after_steps
        self.holding_execution_reward = holding_execution_reward

        if self.always_buy or self.always_sell:
            self.signal_weight = 0

        if self.always_buy and self.always_sell:
            raise AlwaysBuyAndAlwaysSellError(
                "A model cannot always buy and always sell"
            )

        # TODO: add ability to cancel?
        # Action space: [buy, sell, hold], [volume], [price]

        # Define the number of discrete values for each dimension

        if holdable:
            max_side = 3
        else:
            max_side = 2

        num_discrete_values = [
            max_side,
            max_volume + 1,
            1000,
        ]

        # Create the corresponding MultiDiscrete space
        self.action_space = gym.spaces.MultiDiscrete(num_discrete_values)

        # TODO: add outstanding orders?
        # Observation space:
        # Minute in interval
        # Side this interval
        # Volume left in interval
        # data

        num_vars = len(self.trade_data.columns)

        minute_space = gym.spaces.Box(
            low=0, high=interval_length, shape=(1,), dtype=np.int32
        )
        side_space = gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.int32)
        volume_space = gym.spaces.Box(
            low=0, high=max_volume, shape=(1,), dtype=np.int32
        )

        # Assuming num_vars is the number of variables in your data
        data_space = gym.spaces.Box(
            low=-inf, high=inf, shape=(num_vars,), dtype=np.float64
        )

        # Combine individual spaces into a dictionary
        self.observation_space = gym.spaces.Dict(
            {
                "minute": minute_space,
                "side": side_space,
                "volume_left": volume_space,
                "data": data_space,
            }
        )

        # Get the intervals
        if intervals is None:
            self.intervals = data.get_intervals_with_full_bid_and_ask(
                interval_length
            )
        else:
            self.intervals = intervals

        skip_due_to_lags = int(self.num_lags / self.interval_length)

        self.interval_starts_iter = itertools.cycle(
            [interval[0] for interval in self.intervals][skip_due_to_lags:]
        )

        self.observed_minute = next(self.interval_starts_iter)
        self.next_minute_in_interval = self.observed_minute + pd.Timedelta(
            minutes=1
        )

        self.best_bid_anchor = self.trade_data.loc[self.observed_minute][
            "best_bid"
        ]

        not_substracted_columns = [
            col_name
            for col_name in self.trade_data.columns
            if "volume" in col_name
        ]
        not_substracted_columns.append("minutes_in_day")

        self.columns_to_subtract = [
            col_name
            for col_name in self.trade_data.columns
            if col_name not in not_substracted_columns
        ]

        self.minute_in_interval = 1
        self.side_this_interval = 2  # = HOLD
        self.volume_left = max_volume
        self.total_price_in_interval = 0
        self.outstanding_orders = []
        self.inserted_orders = []
        self.executed_orders = []
        self.benchmark_orders = []
        self.interval_low = inf
        self.interval_high = -inf
        self.execution_reward = 0
        self.execution_reward_bm = 0
        self.signal_reward = 0
        self.total_steps = 0

        self.interval_ending = False
        self.interval_start_bid = None
        self.interval_start_ask = None

    def handle_outstanding_orders(self):
        """
        Handles outstanding orders based on the current minute's data.

        This method updates the interval low and high values based on the current minute's data.
        It checks if any orders have been executed and updates the volume left and total price in the interval accordingly.
        If the volume left becomes zero or negative, all outstanding orders are canceled.
        """
        minute_low = self.row_last_minute["low"]
        minute_high = self.row_last_minute["high"]
        minute_volume = self.row_last_minute["volume"]
        minute_bid = self.row_last_minute["best_bid"]
        minute_ask = self.row_last_minute["best_ask"]
        self.interval_low = min(self.interval_low, minute_low, minute_bid)
        self.interval_high = max(self.interval_high, minute_high, minute_ask)

        self.executed_orders = []

        # If volume = 0, this row was interpolated hence no trades happened
        if minute_volume > 0:
            # If buying this interval
            if self.side_this_interval == 0:

                # Check if orders have been executed
                for order in self.outstanding_orders:
                    if order.price >= minute_low:
                        self.volume_left -= order.volume
                        self.total_price_in_interval += (
                            order.price * order.volume
                        )
                        self.executed_orders.append(order)

            # If selling this interval
            elif self.side_this_interval == 1:

                # Check if orders have been executed
                for order in self.outstanding_orders:
                    if order.price <= minute_high:
                        self.volume_left -= order.volume
                        self.total_price_in_interval += (
                            order.price * order.volume
                        )
                        self.executed_orders.append(order)

            self.outstanding_orders = [
                order
                for order in self.outstanding_orders
                if order not in self.executed_orders
            ]

        # Cancel all orders if we have done our volume
        if self.volume_left <= 0 or self.cancel_after_one_minute:
            self.outstanding_orders = []

    def handle_ended_interval(self):
        """
        Handles the end of an interval in the trading environment.

        This method calculates the rewards based on the trading actions taken during the interval.
        It updates the total reward, resets certain variables, and prepares for the next interval.

        Returns:
            None
        """
        self.observed_minute = self.next_minute_in_interval - pd.Timedelta(
            minutes=1
        )

        current_ask = self.row_last_minute["best_ask"]
        current_bid = self.row_last_minute["best_bid"]
        current_mid_price = (current_bid + current_ask) / 2
        current_spread = current_ask - current_bid

        interval_start_mid_price = (
            self.interval_start_ask + self.interval_start_bid
        ) / 2

        if self.side_this_interval == 0:  # buy
            # # Signal reward
            if (
                current_mid_price - interval_start_mid_price
                > current_spread * self.signal_multiplier * self.holdable
            ):
                self.signal_reward = 1
            else:
                self.signal_reward = 0

            # # Execution reward
            # Buy the remaining volume at the market price
            if self.volume_left > 0:
                self.total_price_in_interval += self.volume_left * current_ask

                market_order = Order(
                    OrderType.MARKET_ORDER,
                    OrderSide.BID,
                    self.volume_left,
                    None,
                    self.observed_minute + pd.Timedelta(minutes=1),
                )
                self.inserted_orders.append(market_order)

            # Sell the excess volume at the market price
            if self.volume_left < 0:
                self.total_price_in_interval += self.volume_left * current_bid

                market_order = Order(
                    OrderType.MARKET_ORDER,
                    OrderSide.ASK,
                    self.volume_left,
                    None,
                    self.observed_minute + pd.Timedelta(minutes=1),
                )
                self.inserted_orders.append(market_order)

            high_price = self.interval_high * self.max_volume
            low_price = self.interval_low * self.max_volume

            self.execution_reward = get_execution_reward(
                0, self.total_price_in_interval, low_price, high_price
            )

            self.execution_reward_bm = get_execution_reward(
                0,
                self.interval_start_ask * self.max_volume,
                low_price,
                high_price,
            )

            # Penalize not hitting our max voume or going beyond it
            self.execution_reward -= self.volume_penalizer * abs(
                self.volume_left
            )

        elif self.side_this_interval == 1:  # sell
            # # Signal reward
            if (
                interval_start_mid_price - current_mid_price
                > current_spread * self.signal_multiplier * self.holdable
            ):
                self.signal_reward = 1
            else:
                self.signal_reward = 0

            # # Execution reward
            # Sell the remaining volume at the market price
            if self.volume_left > 0:
                self.total_price_in_interval += self.volume_left * current_bid

                market_order = Order(
                    OrderType.MARKET_ORDER,
                    OrderSide.ASK,
                    self.volume_left,
                    None,
                    self.observed_minute + pd.Timedelta(minutes=1),
                )
                self.inserted_orders.append(market_order)

            # Buy the excess volume at the market price
            if self.volume_left < 0:
                self.total_price_in_interval += self.volume_left * current_ask

                market_order = Order(
                    OrderType.MARKET_ORDER,
                    OrderSide.BID,
                    self.volume_left,
                    None,
                    self.observed_minute + pd.Timedelta(minutes=1),
                )
                self.inserted_orders.append(market_order)

            high_price = self.interval_high * self.max_volume
            low_price = self.interval_low * self.max_volume

            self.execution_reward = get_execution_reward(
                1, self.total_price_in_interval, low_price, high_price
            )

            self.execution_reward_bm = get_execution_reward(
                1,
                self.interval_start_bid * self.max_volume,
                low_price,
                high_price,
            )

            # TODO: check if this does what we want
            # Penalize not hitting our max voume
            self.execution_reward -= self.volume_penalizer * abs(
                self.volume_left
            )

        elif self.side_this_interval == 2:  # hold
            self.execution_reward = (
                self.holding_execution_reward  # TODO: maybe change to something higher? [used to be 0]
            )

            if (
                abs(current_mid_price - interval_start_mid_price)
                < current_spread * self.signal_multiplier
            ):
                self.signal_reward = 1
            else:
                self.signal_reward = 0

        self.reward += (
            self.signal_weight * self.signal_reward
            + self.execution_weight * self.execution_reward
        )

        # Reset things for the new interval
        self.total_price_in_interval = 0
        self.volume_left = self.max_volume
        self.outstanding_orders = []
        self.interval_ending = False
        self.interval_low = inf
        self.interval_high = -inf

    def get_price(self, price_extra):
        """
        Calculates and returns the price based on the current side of the interval.

        Args:
            price_extra (float): The extra price to be added or subtracted.

        Returns:
            float: The calculated price.
        """
        price = 0

        if self.side_this_interval == 0:
            price = self.row_last_minute["best_ask"] - price_extra / 200
        elif self.side_this_interval == 1:
            price = self.row_last_minute["best_bid"] + price_extra / 200

        return round(price, 3)

    def handle_ending_interval(self):
        """
        Handles the ending of an interval in the trading environment.
        Updates the necessary variables and prepares for the next interval.
        """
        self.minute_in_interval = 1
        self.next_minute_in_interval = next(self.interval_starts_iter, None)

        self.interval_ending = True
        self.best_bid_anchor = self.trade_data.loc[
            self.next_minute_in_interval
        ]["best_bid"]

    def step(self, action: ActType):
        """
        Executes a single step in the trading environment.

        Args:
            action (ActType): The action to take in the environment.

        Returns:
            Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]: A tuple containing the new observation space,
            the reward, a flag indicating if the episode is terminated, a flag indicating if the observation is truncated,
            and additional information.
        """
        self.reward = 0

        self.row_last_minute = self.trade_data.loc[self.observed_minute]

        self.handle_outstanding_orders()

        if self.interval_ending:
            self.handle_ended_interval()

        # Unpack action
        side, volume, price_extra = action
        print(self.side_this_interval)

        volume = min(self.volume_left, volume)

        if self.always_buy:
            side = 0
        elif self.always_sell:
            side = 1

        # Update the time
        self.row_last_minute = self.trade_data.loc[self.observed_minute]
        self.observed_minute += pd.Timedelta(minutes=1)

        # Change the side at the start of the interval
        if self.minute_in_interval == 1:
            self.side_this_interval = side
            self.interval_start_ask = self.row_last_minute["best_ask"]
            self.interval_start_bid = self.row_last_minute["best_bid"]

            if self.side_this_interval != 2:
                benchmark_order = Order(
                    OrderType.MARKET_ORDER,
                    side_int_to_orderside[side],
                    self.max_volume,
                    None,
                    self.observed_minute,
                )

                self.benchmark_orders.append(benchmark_order)

        price = self.get_price(price_extra)

        # Insert the new order or immediatly execute
        if (
            self.volume_left > 0
            and self.side_this_interval != 2
            and volume > 0
        ):

            order = Order(
                OrderType.LIMIT_ORDER,
                side_int_to_orderside[self.side_this_interval],
                volume,
                price,
                self.observed_minute,
            )

            self.inserted_orders.append(order)

            if price_extra == 0:
                self.total_price_in_interval += price * volume
                self.volume_left -= volume
                self.execucted_orders.append(order)
            else:
                self.outstanding_orders.append(order)

        # Handle an ending interval or update the minutes in the interval
        if self.minute_in_interval == self.interval_length:
            self.handle_ending_interval()
            volume_to_pass_on = self.max_volume
        else:
            self.minute_in_interval += 1
            volume_to_pass_on = self.volume_left

        # Other things that are returned this step
        data = self.trade_data.loc[self.next_minute_in_interval].copy()
        data[self.columns_to_subtract] -= self.best_bid_anchor

        new_observation_space = {
            "minute": np.array([self.minute_in_interval]),
            "side": np.array([self.side_this_interval]),
            "volume_left": np.array([volume_to_pass_on]),
            "data": data.to_numpy().flatten(),
        }

        info = {
            "time": self.next_minute_in_interval,
            "signal_reward": self.signal_reward,
            "execution_reward": self.execution_reward,
            "execution_reward_bm": self.execution_reward_bm,
            "inserted_orders": self.inserted_orders,
            "executed_orders": self.executed_orders,
            "benchmark_orders": self.benchmark_orders,
        }

        self.signal_reward = 0
        self.execution_reward = 0
        self.execution_reward_bm = 0
        self.inserted_orders = []
        self.execucted_orders = []
        self.benchmark_orders = []

        self.next_minute_in_interval += pd.Timedelta(minutes=1)
        truncated = False

        if self.total_steps == self.stop_after_steps:
            terminated = True
        else:
            terminated = False
            self.total_steps += 1

        return (
            new_observation_space,
            self.reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=""):
        """
        Resets the trading environment to its initial state.

        Args:
            seed (int): The seed value for random number generation.

        Returns:
            tuple: A tuple containing the first observation space and additional information.
        """

        super().reset(seed=seed)

        # Return the first observation
        data = self.trade_data.loc[self.next_minute_in_interval].copy()
        data[self.columns_to_subtract] -= self.best_bid_anchor

        first_observation_space = {
            "minute": np.array([self.minute_in_interval]),
            "side": np.array([self.side_this_interval]),
            "volume_left": np.array([self.volume_left]),
            "data": data.to_numpy().flatten(),
        }

        info = {
            "time": self.next_minute_in_interval,
            "signal_reward": self.signal_reward,
            "execution_reward": self.execution_reward,
            "execution_reward_bm": self.execution_reward_bm,
            "insert_orders": self.inserted_orders,
            "executed_orders": self.executed_orders,
            "benchmark_orders": self.benchmark_orders,
        }

        return first_observation_space, info

    def render(self):
        """
        Renders the environment
        """
        pass

    def close(self):
        """
        Closes the environment
        """
        pass


def get_execution_reward(action, price, low_price, high_price):
    """
    Calculate the execution reward based on the given action, price, low_price, and high_price.

    Parameters:
    - action (int): The action taken (0 for 'buy', 1 for 'sell').
    - price (float): The current price.
    - low_price (float): The lowest price in the range.
    - high_price (float): The highest price in the range.

    Returns:
    - reward (float): The calculated execution reward.

    Raises:
    - ValueError: If the action is not valid (not 0 or 2).
    """
    if abs(high_price - low_price) < 0.001:
        return 0

    # Normalize the price to the range [0, 1]
    normalized_price = (price - low_price) / (high_price - low_price)

    # If the action is 'buy', reward is higher for lower prices
    if action == 0:
        reward = 1 - normalized_price
    # If the action is 'sell', reward is higher for higher prices
    elif action == 1:
        reward = normalized_price
    else:
        raise ValueError("Invalid action")

    # Bound between 0 and 1
    reward = min(1, reward)
    reward = max(0, reward)

    return reward


class BuyEnv(TradingEnv):

    def __init__(self):

        train_data_path = os.path.join(
            os.path.dirname(__file__), "train_data.pkl"
        )

        with open(train_data_path, "rb") as f:
            train_data = pickle.load(f)

        interval_path = os.path.join(
            os.path.dirname(__file__), "train_intervals.pkl"
        )

        with open(interval_path, "rb") as f:
            train_intervals = pickle.load(f)

        num_lags = 20
        interval_length = 5
        max_volume = 20
        holdable = False
        always_buy = True
        cancel_after_one_minute = True
        stop_after_steps = 10000

        super().__init__(
            train_data,
            num_lags,
            interval_length,
            max_volume,
            holdable=holdable,
            always_buy=always_buy,
            cancel_after_one_minute=cancel_after_one_minute,
            intervals=train_intervals,
            stop_after_steps=stop_after_steps,
        )
