from .model_result import ModelResult
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from ..dataclasses import (
    TradingSignals,
    TradingSignal,
    ProcessedData,
    OrderSide,
)
from ..utils import encode
import pandas as pd


class GridSearchSignalResult(ModelResult):
    NO_EXOG = [
        "from_timestamp",
        "to_timestamp",
        "optimal_action",
        "enum_column",
        "optimal_action_encoded",
    ]

    def __init__(
        self,
        grid_search_result: GridSearchCV,
        interval_length: int,
        num_lags: int,
        encoder: LabelEncoder,
    ):
        self.grid_search_result = grid_search_result
        self.interval_length = interval_length
        self.num_lags = num_lags
        self.encoder = encoder

    def get_output(
        self, data: ProcessedData, order_size: int
    ) -> TradingSignals:

        X = data.get_df_for_ml(
            self.interval_length,
            self.num_lags,
            self.encoder,
            fit_encoder=False,
        )

        X = X.dropna()

        encode(X, self.encoder, fit_encoder=False)

        exog = [column for column in X.columns if column not in self.NO_EXOG]

        y_pred = self.grid_search_result.best_estimator_.predict(X[exog])
        print(
            list(y_pred).count(0), list(y_pred).count(1), list(y_pred).count(2)
        )

        decoded_y_pred = self.encoder.inverse_transform(y_pred)

        trading_signals = TradingSignals()

        for index, action in enumerate(decoded_y_pred):
            time = X.index[index]
            match action:
                case "HOLD":
                    continue
                case "BUY":
                    side = OrderSide.BID
                case "SELL":
                    side = OrderSide.ASK

            signal = TradingSignal(
                side, order_size, time + pd.Timedelta(minutes=1)
            )
            trading_signals.append(signal)

        result_dict = {
            "signals": trading_signals,
            "y_pred": y_pred,
            "y_actual": X["optimal_action_encoded"],
        }

        return result_dict

    def summary(self):
        pass
