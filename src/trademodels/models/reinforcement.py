from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

from .model import Model
from ..gym.trading_env import TradingEnv
from ..dataclasses import ProcessedData
from torch import nn


class RLModel(Model):
    """
    Reinforcement Learning Model class.

    Args:
        data (ProcessedData): The processed data used for training the model.
        num_lags (int): The number of lagged features to include in the input data.
        always_buy (bool, optional): Flag indicating whether to always take a buy action. Defaults to False.
        always_sell (bool, optional): Flag indicating whether to always take a sell action. Defaults to False.
    """

    def __init__(
        self,
        data: ProcessedData,
        num_lags: int,
        always_buy: bool = False,
        always_sell: bool = False,
        cancel_after_one_minute: bool = False,
    ):
        """
        Initializes a ReinforcementModel object.

        Args:
            data (ProcessedData): The processed data used for training the model.
            num_lags (int): The number of lagged features to include in the input data.
            always_buy (bool, optional): Flag indicating whether to always take a buy action. Defaults to False.
            always_sell (bool, optional): Flag indicating whether to always take a sell action. Defaults to False.
        """
        self.data = data
        self.num_lags = num_lags
        self.always_buy = always_buy
        self.always_sell = always_sell
        self.cancel_after_one_minute = cancel_after_one_minute

    def train(
        self,
        total_timestamps: int,
        interval_length: int,
        max_volume: int,
        path: str,
        use_saved: bool = False,
        n_epochs: int = 10,
        num_saving: int = 1,
    ):
        """
        Trains the reinforcement learning model.

        Args:
            total_timestamps (int): Total number of timestamps to train for.
            interval_length (int): Length of each interval.
            max_volume (int): Maximum volume.
            path (str): Path to save the trained model.
            use_saved (bool, optional): Whether to use a saved model for training. Defaults to False.
            n_epochs (int, optional): Number of epochs for training. Defaults to 10.
            num_saving (int, optional): Number of times to save the model during training. Defaults to 1.
        """
        # Parallel environments
        vec_env = make_vec_env(
            self.env_fn(interval_length, max_volume), n_envs=1
        )

        # Set up the recurrent PPO model
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            vec_env,
            n_steps=128,
            n_epochs=n_epochs,
            verbose=1,
            policy_kwargs={
                "n_lstm_layers": 2,
                "lstm_hidden_size": 128,
                "activation_fn": nn.ReLU,
            },
            ent_coef=0.0033426106829936,
            learning_rate=0.00021845518212898,
            clip_range=0.3,
            batch_size=16,
        )

        timestamps_per_block = int(total_timestamps / num_saving)

        checkpoint_callback = CheckpointCallback(
            save_freq=timestamps_per_block,
            save_path=f"./{path}/",
            name_prefix=path,
        )

        callbacks = CallbackList(
            # [checkpoint_callback, ResetLSTMStateCallback()] # NOTE: resetting gives worse performance
            [checkpoint_callback]
        )

        if use_saved:
            model.set_parameters(path)

        model.learn(
            total_timesteps=total_timestamps,
            progress_bar=False,
            callback=callbacks,
        )
        model.save(path)

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


class ResetLSTMStateCallback(BaseCallback):
    """
    Callback class to reset the LSTM state of a RL model every few steps.

    Attributes:
        reset_every (int): The number of steps after which the LSTM state should be reset.

    Methods:
        _on_step(): Callback method called after each training step.
    """

    reset_every = 5

    def __init__(self):
        super(ResetLSTMStateCallback, self).__init__()

    def _on_step(self):
        """
        Callback method called after each training step.

        If the number of timesteps is a multiple of `reset_every`, the LSTM state of the RL model is reset.
        """
        if self.num_timesteps % self.reset_every == 0:
            # Assuming 'model' is your RL model with an LSTM layer
            self.model.last_lstm_states = None
        return True
