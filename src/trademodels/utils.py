import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def get_result_dict_from_result_tuple(result_tuple) -> dict:
    """
    Converts a result tuple into a dictionary with named fields.

    Args:
        result_tuple (tuple): The result tuple to convert.

    Returns:
        dict: A dictionary with named fields corresponding to the result tuple.

    Example:
        result_tuple = (10, 0.05, 8, 0.03, 'buy', 'sell', [order1, order2])
        result_dict = get_result_dict_from_result_tuple(result_tuple)
        # result_dict will be:
        # {
        #     'trades': 10,
        #     'pcts_off': 0.05,
        #     'trades_bm': 8,
        #     'pcts_off_bm': 0.03,
        #     'optimal_sides': 'buy',
        #     'chosen_sides': 'sell',
        #     'executed_orders': [order1, order2]
        # }
    """
    result_dict = {}

    index_to_name = {
        1: "trades",
        2: "pcts_off",
        3: "trades_bm",
        4: "pcts_off_bm",
        5: "optimal_sides",
        6: "chosen_sides",
        7: "executed_orders",
    }

    for index, field in enumerate(result_tuple):
        name = index_to_name[index]

        result_dict[name] = field

    return result_dict


def get_aggresiveness_from_trades(trades: list, data) -> list:
    """
    Calculate the aggressiveness of trades based on the order type, side, and current market prices.

    Args:
        trades (list[Order]): List of Order objects representing the trades.
        data (ProcessedData): ProcessedData object containing the market data.

    Returns:
        list: List of tuples containing the trade time and corresponding aggressiveness.

    """
    from trademodels.dataclasses import OrderType, OrderSide

    df_to_time_index = data.set_index("to_timestamp")

    output = []

    for order in trades:

        if order.type == OrderType.MARKET_ORDER:
            continue

        if order.side == OrderSide.BID:
            current_ask = df_to_time_index.loc[order.time]["best_ask"]
            aggresiveness = current_ask - order.price

        elif order.side == OrderSide.ASK:
            current_bid = df_to_time_index.loc[order.time]["best_bid"]
            aggresiveness = order.price - current_bid

        output.append((order.time, aggresiveness))

    return output


def calculate_percentage_in_range(data, start, end):
    """
    Calculates the percentage of values in the given data that fall within the specified range.

    Args:
        data (list): A list of numeric values.
        start (float): The lower bound of the range.
        end (float): The upper bound of the range.

    Returns:
        float: The percentage of values in the data that fall within the specified range.
    """
    return sum(1 for value in data if start <= value <= end) / len(data) * 100


def moving_window_percentage(data, window_size, start, end):
    """
    Calculate the percentage of values in a moving window that fall within a specified range.

    Args:
        data (list): The input data.
        window_size (int): The size of the moving window.
        start (float): The lower bound of the range.
        end (float): The upper bound of the range.

    Returns:
        list: A list of percentages, where each percentage represents the proportion of values in the moving window
              that fall within the specified range.
    """
    percentages = []
    for i in range(len(data) - window_size + 1):
        window_data = data[i : i + window_size]
        percentage = calculate_percentage_in_range(window_data, start, end)
        percentages.append(percentage)
    return percentages


def plot_aggresiveness_over_time(
    times: list[pd.Timestamp],
    agg: list[float],
    window_size: int,
    save_path: str = None,
):
    """
    Plots the aggressiveness over time.

    Args:
        times (list[pd.Timestamp]): List of timestamps.
        agg (list[float]): List of aggressiveness values.
        window_size (int): Size of the moving window.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # start_end_ranges = [(0, 5 / 3), (5 / 3, 10 / 3), (10 / 3, 15 / 3)]
    start_end_ranges = [(0, 1.67), (1.67, 3.33), (3.33, 5)]

    plt.figure(figsize=(10, 6))

    for start, end in start_end_ranges:
        percentages = moving_window_percentage(agg, window_size, start, end)
        plt.plot(
            times[: len(percentages)], percentages, label=f"{start}-{end}"
        )

    plt.xlabel("Date")
    plt.ylabel("Percentage")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def encode(trimmed_df, encoder, fit_encoder: bool = True):
    # Encode the 'optimal_action' column
    trimmed_df["enum_column"] = trimmed_df["optimal_action"].apply(
        lambda x: x.name
    )  # Convert enum to its name

    if fit_encoder:
        trimmed_df["optimal_action_encoded"] = encoder.fit_transform(
            trimmed_df["enum_column"]
        )
    else:
        trimmed_df["optimal_action_encoded"] = encoder.transform(
            trimmed_df["enum_column"]
        )
    return encoder


def get_ml_encoder() -> LabelEncoder:
    # Create LabelEncoder
    encoder = LabelEncoder()

    # Define the mapping between labels and encoded values
    label_mapping = {"BUY": 0, "SELL": 1, "HOLD": 2}

    # Fit the encoder with the defined labels
    encoder.fit(list(label_mapping.keys()))

    # Manually specify the encoding
    encoder.classes_ = list(label_mapping.keys())

    encoder = OrdinalEncoder(categories=[["BUY", "SELL", "HOLD"]])

    return encoder
