from src.environment import NumpyStockTradingEnv, utils
from src.preprocessing import df_to_numpy

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.stablebaselines3 import models as finrl_models
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR




parser = ArgumentParser(
            prog='Train script')

parser.add_argument('--exp-name', type=str, required=True,
                    help='experiment name for training session')

parser.add_argument('--n-env', type=int, required=True,
                    help='number of parallel environments to use')


if __name__ == "__main__":
    args = parser.parse_args()
    exp_name = args.exp_name
    n_envs = args.n_env

    train = pd.read_csv('prepared_csv/russel1000_train_data.csv')
    train = train.set_index(train.columns[0])
    train.index.names = ['']


    tic_list = sorted(train['tic'].unique().tolist())
    INDICATORS = [
        'day',
        'macd',
        'boll_ub',
        'boll_lb',
        'rsi_30',
        'cci_30',
        'dx_30',
        'close_30_sma',
        'close_60_sma',
        'vix',
    ]
    price_array, tech_array, turbulence_array, dates = df_to_numpy(
        train, 
        tic_list, 
        tech_indicator_list=INDICATORS, 
        price_col="close", 
        turbulence_col="turbulence"
    )

    env_kwargs = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": 0,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": None,
        "stock_dim": len(tic_list),
        "tech_indicator_list": INDICATORS,
        "action_space": len(tic_list),
        "reward_scaling": 1e-4,
        "allow_negative_cash": False,
        "max_stock_holding": np.full(len(tic_list), 1e9),}

    make_env = lambda: NumpyStockTradingEnv(**env_kwargs)
    env = SubprocVecEnv([make_env] * n_envs)

    finrl_models.TensorboardCallback._on_rollout_end = utils.safe_on_rollout_end
    finrl_models.DRLAgent.train_model = staticmethod(utils.safe_train_model)

    agent = DRLAgent(env=env)
    SAC_PARAMS = {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "buffer_size": 140_000,
        "learning_starts": 5_000,
        "gamma": 0.99,
        "tau": 0.005,
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)


    tmp_path = os.path.join(RESULTS_DIR, 'sac')
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

    trained_sac = agent.train_model(
        model=model_sac, 
        tb_log_name=exp_name,
        total_timesteps=200_000
    )   

    trained_sac.save(TRAINED_MODEL_DIR + f"/{exp_name}")

