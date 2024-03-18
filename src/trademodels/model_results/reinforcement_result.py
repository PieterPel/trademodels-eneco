from .model_result import ModelResult
from ..dataclasses import (
    ProcessedData,
    Order,
    OrderType,
    OrderSide,
    PctOff,
    RLResults,
)
from stable_baselines3.common.env_util import make_vec_env
from ..gym.trading_env import TradingEnv

from ..errors import AlwaysBuyAndAlwaysSellError
from typing import List
import os

import pandas as pd
import numpy as np
from math import inf
from stable_baselines3 import PPO
import dill as pickle


class RLModelResult(ModelResult):
    """
    Represents the result of a reinforcement learning model.

    Attributes:
        num_lags (int): The number of lagged observations to include as input features.
        interval_length (int): The length of each trading interval in minutes.
        max_volume (int): The maximum volume of shares that can be traded in each interval.
        always_buy (bool): Whether the model should always buy.
        always_sell (bool): Whether the model should always sell.
        model (PPO): The trained reinforcement learning model.

    Methods:
        __init__: Initializes a ReinforcementResult object.
        get_output: Returns the output of the model given the input data.

    Raises:
        AlwaysBuyAndAlwaysSellError: If both always_buy and always_sell are set to True.
    """

    def __init__(
        self,
        path_to_saved_zip: str,
        num_lags: int,
        interval_length: int,
        max_volume: int,
        always_buy: bool = False,
        always_sell: bool = False,
        cancel_after_one_minute: bool = False,
        holdable: bool = False,
    ) -> None:
        """
        Initializes a ReinforcementResult object.

        Args:
            path_to_saved_zip (str): The path to the saved zip file containing the trained model.
            num_lags (int): The number of lagged observations to include as input features.
            interval_length (int): The length of each trading interval in minutes.
            max_volume (int): The maximum volume of shares that can be traded in each interval.
            always_buy (bool, optional): Whether the model should always buy. Defaults to False.
            always_sell (bool, optional): Whether the model should always sell. Defaults to False.

        Raises:
            AlwaysBuyAndAlwaysSellError: If both always_buy and always_sell are set to True.
        """
        self.num_lags = num_lags
        self.interval_length = interval_length
        self.max_volume = max_volume
        self.always_buy = always_buy
        self.always_sell = always_sell
        self.cancel_after_one_minute = cancel_after_one_minute
        self.holdable = holdable
        self.model = PPO.load(path_to_saved_zip)

        if always_buy and always_sell:
            raise AlwaysBuyAndAlwaysSellError(
                "A model cannot always buy and always sell"
            )

    # NOTE: Orders are on the to_time
    def get_output(self, data: ProcessedData) -> List[Order]:
        """Returns the trades made by the model for some given data

        Args:
            data (ProcessedData): The ProcessedData that contains the trading data is used

        Returns:
            List[Order]: A list of the orders put in the order book
        """
        trade_data = data.get_df_for_rl(self.num_lags)
        intervals = data.get_intervals_with_full_bid_and_ask(
            self.interval_length,
        )

        # Skip due to missing lags:
        to_skip = int(self.num_lags / self.interval_length) + 1

        self.interval_starts = [
            interval[0] for interval in intervals[to_skip:]
        ]
        orders = []
        pcts_off = []
        benchmark_orders = []
        benchmark_pcts_off = []
        optimal_sides = []
        chosen_sides = []
        executed_orders = []

        counter = 0
        side_this_interval = 2  # hold
        total_price = 0
        bm_total_price = 0
        interval_start_ask = 0
        interval_start_bid = 0

        interval_ending = False
        outstanding_orders = []
        interval_low = inf
        interval_high = -inf

        # For the first interval
        volume_left = self.max_volume
        check_time = self.interval_starts[0]

        best_bid_anchor = trade_data.loc[check_time]["best_bid"]

        columns_to_subtract = trade_data.columns[
            (trade_data.columns != "minutes_in_day")
            & ("volume" in trade_data.columns)
        ]

        # Cycle over all intervals
        for from_time in self.interval_starts:
            # # At the start of the interval
            # trade_data[from_time] has the current bid and ask, alongside the OHLCV data of the 'previous minute'
            minute_in_interval = 1

            counter += 1
            if counter == 1000:
                counter = 0
                print(from_time)

            data = trade_data.loc[from_time].copy()
            data[columns_to_subtract] -= best_bid_anchor

            lstm_states = None
            num_envs = 1
            # Episode start signals are used to reset the lstm states
            episode_starts = np.ones((num_envs,), dtype=bool)

            observation_space = {
                "minute": np.array([minute_in_interval]),
                "side": np.array([side_this_interval]),
                "volume_left": np.array([volume_left]),
                "data": data.to_numpy().flatten(),
            }

            action, lstm_states = self.model.predict(
                observation_space,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            executed_orders_minute = []

            while minute_in_interval <= self.interval_length:
                # # During the interval

                row = trade_data.loc[check_time]
                # Update the interval low and high
                if minute_in_interval > 1:
                    interval_low = min(interval_low, row["low"])
                    interval_high = max(interval_high, row["high"])

                # Check if orders have executed #TODO: make less spaghetti code
                if outstanding_orders != [] and row["volume"] > 0:

                    for order in outstanding_orders:
                        if side_this_interval == 0:
                            if order.price >= row["low"]:
                                volume_left -= order.volume
                                total_price += order.volume * order.price
                                executed_orders_minute.append(order)

                        if side_this_interval == 1:
                            if order.price <= row["high"]:
                                volume_left -= order.volume
                                total_price += order.volume * order.price
                                executed_orders_minute.append(order)

                    outstanding_orders = [
                        order
                        for order in outstanding_orders
                        if order not in executed_orders_minute
                    ]
                    executed_orders.extend(executed_orders_minute)

                # Cancel outstanding orders if we have done all our volume
                if volume_left <= 0 or self.cancel_after_one_minute:
                    outstanding_orders = []

                if interval_ending:
                    interval_ending = False
                    to_time = check_time + pd.Timedelta(minutes=1)

                    # Store the optimal side taken this interval, alongside the one taken
                    current_ask = trade_data.loc[check_time]["best_ask"]
                    current_bid = trade_data.loc[check_time]["best_bid"]
                    current_mid_price = (current_bid + current_ask) / 2
                    current_spread = current_ask - current_bid

                    interval_start_mid_price = (
                        interval_start_ask + interval_start_bid
                    ) / 2

                    mid_change = current_mid_price - interval_start_mid_price

                    if mid_change > current_spread * self.holdable:
                        optimal_side = 0
                    elif mid_change < -1 * current_spread * self.holdable:
                        optimal_side = 1
                    else:
                        optimal_side = 2

                    optimal_sides.append(optimal_side)
                    chosen_sides.append(side_this_interval)

                    # Execute the remaining volume at the current price
                    if side_this_interval == 0 and volume_left > 0:
                        order = Order(
                            OrderType.MARKET_ORDER,
                            OrderSide.BID,
                            volume_left,
                            None,
                            to_time,
                        )
                        orders.append(order)
                        price = trade_data.loc[check_time]["best_ask"]
                        total_price += volume_left * price

                    # Execute the excess volume at the current price
                    if side_this_interval == 0 and volume_left < 0:
                        order = Order(
                            OrderType.MARKET_ORDER,
                            OrderSide.ASK,
                            -1 * volume_left,
                            None,
                            to_time,
                        )
                        orders.append(order)
                        price = trade_data.loc[check_time]["best_bid"]
                        total_price += volume_left * price

                    # Execute the remaining volume at the current price
                    elif side_this_interval == 1 and volume_left > 0:
                        order = Order(
                            OrderType.MARKET_ORDER,
                            OrderSide.ASK,
                            volume_left,
                            None,
                            to_time,
                        )
                        orders.append(order)
                        price = trade_data.loc[check_time]["best_bid"]
                        total_price += volume_left * price

                    # Execute the excess volume at the current price
                    elif side_this_interval == 1 and volume_left < 0:
                        order = Order(
                            OrderType.MARKET_ORDER,
                            OrderSide.BID,
                            -1 * volume_left,
                            None,
                            to_time,
                        )
                        orders.append(order)
                        price = trade_data.loc[check_time]["best_ask"]
                        total_price += volume_left * price

                    # Get the percentage off the best possible price
                    if side_this_interval == 0:
                        # Calculate the best price possible
                        best_price = interval_low * self.max_volume

                        pct_off = 100 * (total_price - best_price) / best_price
                        pct_off_bm = (
                            100 * (bm_total_price - best_price) / best_price
                        )

                    elif side_this_interval == 1:
                        # Calculate the best price possible
                        best_price = interval_high * self.max_volume

                        pct_off = 100 * (best_price - total_price) / best_price
                        pct_off_bm = (
                            100 * (best_price - bm_total_price) / best_price
                        )

                    if side_this_interval in [0, 1]:

                        pcts_off.append(PctOff(check_time, pct_off))
                        benchmark_pcts_off.append(
                            PctOff(check_time, pct_off_bm)
                        )

                    total_price = 0
                    outstanding_orders = []
                    check_time = from_time

                # Unpack action, make sure that they satisfy our constraints
                side, volume, price_extra = action

                volume = min(volume_left, volume)

                if self.always_buy:
                    side = 0
                elif self.always_sell:
                    side = 1

                to_time = check_time + pd.Timedelta(minutes=1)

                # On first minute decide side
                if minute_in_interval == 1:
                    side_this_interval = side
                    volume_left = self.max_volume
                    total_price = 0
                    interval_low = inf
                    interval_high = -inf

                    interval_start_ask = trade_data.loc[check_time]["best_ask"]
                    interval_start_bid = trade_data.loc[check_time]["best_bid"]

                    # Store the data used for the benchmark
                    if side_this_interval == 0:
                        bm_side = OrderSide.BID
                        bm_price = interval_start_ask

                    elif side_this_interval == 1:
                        bm_side = OrderSide.ASK
                        bm_price = interval_start_bid

                    if side_this_interval in [0, 1]:
                        bm_order = order = Order(
                            OrderType.MARKET_ORDER,
                            bm_side,
                            self.max_volume,
                            None,
                            to_time,
                        )

                        bm_total_price = bm_price * self.max_volume

                        benchmark_orders.append(bm_order)

                # Get the price implied by the action
                if side_this_interval == 0:
                    price = round(
                        trade_data.loc[check_time]["best_ask"]
                        - price_extra / 200,
                        3,
                    )
                elif side_this_interval == 1:
                    price = round(
                        trade_data.loc[check_time]["best_bid"]
                        + price_extra / 200,
                        3,
                    )

                # Turn action into order
                if side_this_interval == 0 and volume_left > 0 and volume > 0:

                    order = Order(
                        OrderType.LIMIT_ORDER,
                        OrderSide.BID,
                        volume,
                        price,
                        to_time,
                    )
                    orders.append(order)

                    if price_extra == 0:
                        volume_left -= volume
                        total_price += volume * price
                        executed_orders.append(order)
                    else:
                        outstanding_orders.append(order)

                elif (
                    side_this_interval == 1 and volume_left > 0 and volume > 0
                ):
                    order = Order(
                        OrderType.LIMIT_ORDER,
                        OrderSide.ASK,
                        volume,
                        price,
                        to_time,
                    )

                    orders.append(order)

                    if price_extra == 0:
                        volume_left -= volume
                        total_price += volume * price
                        executed_orders.append(order)
                    else:
                        outstanding_orders.append(order)

                # On minute 5 the interval ends
                if minute_in_interval == self.interval_length:
                    interval_ending = True
                    # episode_starts = np.ones((num_envs,), dtype=bool) # NOTE removed because we dont reset the states

                # Increment the minutes
                check_time += pd.Timedelta(minutes=1)
                minute_in_interval += 1
                data = trade_data.loc[check_time].copy()
                data[columns_to_subtract] -= best_bid_anchor

                # Get new observation and new action
                observation_space = {
                    "minute": np.array([minute_in_interval]),
                    "side": np.array([side_this_interval]),
                    "volume_left": np.array([volume_left]),
                    "data": data.to_numpy().flatten(),
                }

                episode_starts = np.zeros((num_envs,), dtype=bool)

                if not interval_ending:

                    action, lstm_states = self.model.predict(
                        observation_space,
                        state=lstm_states,
                        deterministic=True,
                    )

        return (
            orders,
            pcts_off,
            benchmark_orders,
            benchmark_pcts_off,
            optimal_sides,
            chosen_sides,
            executed_orders,
        )

    def summary(self):
        pass


class RLModelResultFolder(ModelResult):
    """
    Represents a folder containing reinforcement learning model results.

    Args:
        path_to_saved_dir (str): The path to the directory where the result will be saved.
        num_lags (int): The number of lagged observations to consider in the model.
        interval_length (int): The length of each trading interval.
        max_volume (int): The maximum volume that can be traded in each interval.
        always_buy (bool, optional): Whether to always execute a buy action. Defaults to False.
        always_sell (bool, optional): Whether to always execute a sell action. Defaults to False.

    Attributes:
        num_lags (int): The number of lagged observations to consider in the model.
        interval_length (int): The length of each trading interval.
        max_volume (int): The maximum volume that can be traded in each interval.
        always_buy (bool): Whether to always execute a buy action.
        always_sell (bool): Whether to always execute a sell action.
        path_to_saved_dir (str): The path to the directory where the result will be saved.

    Methods:
        get_output(data: ProcessedData) -> None: Retrieves the output of the reinforcement learning model for the given processed data.
        summary() -> None: Provides a summary of the RLModelResultFolder object.
    """

    def __init__(
        self,
        path_to_saved_dir: str,
        num_lags: int,
        interval_length: int,
        max_volume: int,
        always_buy: bool = False,
        always_sell: bool = False,
        cancel_after_one_minute: bool = False,
    ) -> None:
        """
        Initializes a ReinforcementResult object.

        Args:
            path_to_saved_dir (str): The path to the directory where the result will be saved.
            num_lags (int): The number of lagged observations to consider in the model.
            interval_length (int): The length of each trading interval.
            max_volume (int): The maximum volume that can be traded in each interval.
            always_buy (bool, optional): Whether to always execute a buy action. Defaults to False.
            always_sell (bool, optional): Whether to always execute a sell action. Defaults to False.
        """
        self.num_lags = num_lags
        self.interval_length = interval_length
        self.max_volume = max_volume
        self.always_buy = always_buy
        self.always_sell = always_sell
        self.path_to_saved_dir = path_to_saved_dir
        self.cancel_after_one_minute = cancel_after_one_minute

    def get_output(self, data: ProcessedData):

        base_model_path = os.path.join(self.path_to_saved_dir, "models")

        output_folder_path = os.path.join(self.path_to_saved_dir, "output")

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        models: list = os.listdir(base_model_path)

        for model in models:

            model = model.split(".", 1)[0]

            print(f"Now getting the results of {model}")

            model_path = f"{base_model_path}/{model}"

            model_result = RLModelResultAlt(
                model_path,
                self.num_lags,
                self.interval_length,
                self.max_volume,
                self.always_buy,
                self.always_sell,
                self.cancel_after_one_minute,
            )

            model_output = model_result.get_output(data)

            output_path = os.path.join(output_folder_path, model)

            with open(f"{output_path}.pkl", "wb") as f:
                pickle.dump(model_output, f)

    def summary(self):
        pass


def get_results_from_output_folder(path_to_output_folder: str) -> dict:
    """
    Retrieves results from the specified output folder.

    Args:
        path_to_output_folder (str): The path to the output folder containing result pickles.

    Returns:
        dict: A dictionary mapping the name of each result pickle to its corresponding result object.
    """
    result_pickles: list = os.listdir(path_to_output_folder)

    results = {}

    for pkl in result_pickles:

        output_path = os.path.join(path_to_output_folder, pkl)

        with open(output_path, "rb") as f:
            result = pickle.load(f)

        name = pkl.split(".", 1)[0]

        results[name] = result

    return results


def get_pct_summary_from_output_folder(
    path_to_output_folder: str,
) -> pd.DataFrame:
    """
    Calculate the percentage summary from the output folder.

    Args:
        path_to_output_folder (str): The path to the output folder.

    Returns:
        pd.DataFrame: The percentage summary dataframe.
    """

    df = pd.DataFrame()

    pct_off_dict = {}

    results = get_results_from_output_folder(path_to_output_folder)

    for name, result in results.items():

        pct_offs = [x.pct_off for x in result[1]]

        pct_off_description = pd.Series(pct_offs).describe(
            percentiles=[0.9, 0.925, 0.95, 0.975]
        )

        pct_off_dict[name] = pct_off_description

        # execution_reward = [reward for reward in result[3] if reward != 0]

        # execution_reward_description = pd.Series(execution_reward).describe()

        # execution_reward_dict[name] = execution_reward_description

    # Get the benchmark too
    benchmark_pct_off = [x.pct_off for x in result[3]]

    benchmark_pct_off_description = pd.Series(benchmark_pct_off).describe()

    pct_off_dict["benchmark"] = benchmark_pct_off_description

    # Assign the dictionary to the dataframe
    df = df.assign(**pct_off_dict)

    # Sort columns on number of steps
    sorted_columns = sorted(
        df.columns[:-1], key=lambda x: int(x.split("_")[-2])
    )

    df = df[["benchmark"] + sorted_columns]

    return df


def get_reward_summary_from_output_folder(
    path_to_output_folder: str, threshold: float
) -> pd.DataFrame:
    """
    Calculate the reward summary from the output folder.

    Args:
        path_to_output_folder (str): The path to the output folder.

    Returns:
        pd.DataFrame: The reward summary dataframe.
    """

    df = pd.DataFrame()

    execution_reward_dict = {}

    results = get_results_from_output_folder(path_to_output_folder)

    for name, result in results.items():

        execution_reward = [
            reward for reward in result[3] if reward > threshold
        ]

        execution_reward_description = pd.Series(execution_reward).describe(
            percentiles=[0.9, 0.925, 0.95, 0.975]
        )

        execution_reward_dict[name] = execution_reward_description

    benchmark_execution_reward = [
        reward for reward in result[4] if reward > threshold
    ]

    benchmark_execution_reward_description = pd.Series(
        benchmark_execution_reward
    ).describe(percentiles=[0.9, 0.925, 0.95, 0.975])

    execution_reward_dict["benchmark"] = benchmark_execution_reward_description

    # Assign the dictionary to the dataframe
    df = df.assign(**execution_reward_dict)

    # Sort columns on number of steps
    sorted_columns = sorted(
        df.columns[:-1], key=lambda x: int(x.split("_")[-2])
    )

    df = df[["benchmark"] + sorted_columns]

    return df


class RLModelResultAlt(RLModelResult):

    def get_output(
        self, data: ProcessedData, num_envs: int = 1, reset_lstm: bool = False
    ):
        self.data = data

        vec_env = make_vec_env(
            self.env_fn(self.interval_length, self.max_volume), n_envs=num_envs
        )

        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)

        inserted_orders = []
        executed_limit_orders = []

        signal_rewards = []
        execution_rewards = []
        execution_rewards_bm = []
        benchmark_orders = []

        times = []

        counter = 0

        while True:
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )

            (obs, _, _, info) = vec_env.step(action)

            info = info[0]

            if counter == 0:
                first_time = info["time"]
            elif info["time"] == first_time:
                break

            if reset_lstm and obs["minute"] == self.interval_length:
                episode_starts = np.ones((num_envs,), dtype=bool)
            else:
                episode_starts = np.zeros((num_envs,), dtype=bool)

            inserted_orders.extend(info["inserted_orders"])
            executed_limit_orders.extend(info["executed_orders"])
            benchmark_orders.extend(info["benchmark_orders"])
            signal_rewards.append(info["signal_reward"])
            execution_rewards.append(info["execution_reward"])
            execution_rewards_bm.append(info["execution_reward_bm"])
            times.append(info["time"])

            if counter == 5000:
                print(info["time"])
                counter = 0

            counter += 1

        return RLResults(
            inserted_orders,
            executed_limit_orders,
            signal_rewards,
            execution_rewards,
            execution_rewards_bm,
            benchmark_orders,
            times,
        )

    def env_fn(self, interval_length: int, max_volume: int):
        """
        Returns a function that creates a TradingEnv object with the given parameters.

        Args:
            interval_length (int): The length of each trading interval.
            max_volume (int): The maximum volume allowed for trading.

        Returns:
            callable: A function that creates a TradingEnv object.
        """

        def env_fn_inner():
            return TradingEnv(
                self.data,
                self.num_lags,
                interval_length,
                max_volume,
                signal_weight=1,
                execution_weight=1,
                holdable=False,
                always_buy=self.always_buy,
                always_sell=self.always_sell,
                cancel_after_one_minute=self.cancel_after_one_minute,
            )

        return env_fn_inner
