class NonMatchingIntervalError(Exception):
    """Raised if the intervals of a trading signal
    model and an execution model do not match"""


class AlwaysBuyAndAlwaysSellError(Exception):
    """
    Raised if a model is specified that always buys and always sells
    """
