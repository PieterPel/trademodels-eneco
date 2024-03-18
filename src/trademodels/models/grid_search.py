from .model import TradingSignalModel
from ..model_results.grid_search_result import GridSearchSignalResult
from ..dataclasses import ProcessedData
from ..utils import encode
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class GridSearchSignalModel(TradingSignalModel):
    NO_EXOG = [
        "from_timestamp",
        "to_timestamp",
        "optimal_action",
        "enum_column",
        "optimal_action_encoded",
    ]

    def __init__(
        self, data: ProcessedData, num_lags: int, grid_search: GridSearchCV
    ):
        self.data = data
        self.num_lags = num_lags
        self.grid_search = grid_search

    def train(
        self, interval_length: int, holdable: bool = False
    ) -> GridSearchSignalResult:
        encoder = LabelEncoder()

        data = self.data.get_df_for_ml(
            interval_length,
            self.num_lags,
            encoder,
            holdable=holdable,
            fit_encoder=True,
            keep_no_price_change=False,
        )

        encoder = encode(
            data, encoder, fit_encoder=True
        )  # NOTE check if working

        exog = [
            column for column in data.columns if column not in self.NO_EXOG
        ]

        # TODO: probably dont need this function
        X_train, _, y_train, _ = train_test_split(
            data[exog],
            data["optimal_action_encoded"],
            test_size=1,
            shuffle=False,
        )

        self.grid_search.fit(X_train, y_train)
        # print(f"Encoder classes: {encoder.classes_}")
        return GridSearchSignalResult(
            self.grid_search, interval_length, self.num_lags, encoder
        )
