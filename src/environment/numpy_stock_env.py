import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Optional, Sequence, Union, Dict
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque
import pandas as pd


def _to_array_or_broadcast(x, length, dtype=np.float32):
    if np.isscalar(x):
        return np.full(length, float(x), dtype=dtype)
    
    arr = np.array(x, dtype=dtype)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length} or scalar for cost lists; got {arr.shape}")
    
    return arr


class NumpyStockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_array: np.ndarray,              # (T, N)
        tech_array: np.ndarray,               # (T, N, F) or None
        turbulence_array: Optional[np.ndarray] = None,  # (T,)
        hmax: int = 100,
        initial_amount: float = 1_000_000,
        num_stock_shares: Union[float, Sequence[float]] = 0,
        buy_cost_pct: Union[float, Sequence[float]] = 0.001,
        sell_cost_pct: Union[float, Sequence[float]] = 0.001,
        state_space: Optional[int] = None,
        stock_dim: Optional[int] = None,
        tech_indicator_list: Optional[Sequence[str]] = None,
        action_space: Optional[int] = None,
        reward_scaling: float = 1e-4,
        allow_negative_cash: bool = False,
        max_stock_holding: Optional[Union[float, Sequence[float]]] = None,
        risk_indicator_col: Optional[str] = None,
        dates: Optional[Sequence] = None,
        turbulence_penalty_coef: float = 0.5,
        turbulence_normalization: str = "max",  # "max" or "zscore" or "none"
    ):
        super().__init__()

        # --- core arrays ---
        self.prices = price_array.astype(np.float32)
        self.T, self.N = self.prices.shape

        if tech_array is None:
            self.tech = np.zeros((self.T, self.N, 0), dtype=np.float32)
            self.F = 0
        else:
            assert tech_array.ndim == 3, "tech_array must be shape (T,N,F)"
            self.tech = tech_array.astype(np.float32)
            self.F = self.tech.shape[2]

        if turbulence_array is None:
            self.turbulence = np.zeros(self.T, dtype=np.float32)
        else:
            self.turbulence = turbulence_array.astype(np.float32)
            assert len(self.turbulence) == self.T

        # --- config ---
        self.hmax = int(hmax)
        self.initial_amount = float(initial_amount)
        self.reward_scaling = float(reward_scaling)
        self.allow_negative_cash = bool(allow_negative_cash)

        self.stock_dim = self.N if stock_dim is None else int(stock_dim)
        assert self.stock_dim == self.N, "stock_dim mismatch with price_array"

        if np.isscalar(num_stock_shares):
            self.init_holdings = np.full(self.N, float(num_stock_shares), dtype=np.float32)
        else:
            arr = np.array(num_stock_shares, dtype=np.float32)
            assert arr.shape[0] == self.N
            self.init_holdings = arr

        self.buy_cost_pct = _to_array_or_broadcast(buy_cost_pct, self.N)
        self.sell_cost_pct = _to_array_or_broadcast(sell_cost_pct, self.N)

        if max_stock_holding is None:
            self.max_stock_holding = np.full(self.N, np.inf, dtype=np.float32)
        else:
            self.max_stock_holding = _to_array_or_broadcast(max_stock_holding, self.N)

        self.action_dim = self.N if action_space is None else int(action_space)
        assert self.action_dim == self.N, "action_space mismatch with price_array"
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        obs_len = 1 + self.N + self.N + (self.N * self.F) + 1
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.risk_indicator_col = risk_indicator_col

        self.dates = np.array(dates) if dates is not None else None

        # memories (kept for logging)
        self.date_memory = []
        self.asset_memory = []
        self.actions_memory = []

        self.last_action = np.zeros(self.N, dtype=np.int32)

        self.tech_indicator_list = list(tech_indicator_list) if tech_indicator_list else []
        self.state_space = state_space if state_space is not None else obs_len

        # === ADDED: turbulence reward params & normalization stats ===
        self.turbulence_penalty_coef = float(turbulence_penalty_coef)
        if turbulence_normalization not in ("max", "zscore", "none"):
            raise ValueError("turbulence_normalization must be 'max','zscore',or 'none'")
        self.turbulence_normalization = turbulence_normalization

        # compute simple normalization stats (cheap, done once)
        if self.turbulence.size > 0:
            self._turb_max = float(np.max(self.turbulence)) if np.max(self.turbulence) > 0 else 1.0
            self._turb_mean = float(np.mean(self.turbulence))
            self._turb_std = float(np.std(self.turbulence)) if np.std(self.turbulence) > 0 else 1.0
        else:
            self._turb_max = 1.0
            self._turb_mean = 0.0
            self._turb_std = 1.0

        # init store
        self.reset_internal_store()

    def reset_internal_store(self):
        self.day = 0
        self.cash = float(self.initial_amount)
        self.holdings = self.init_holdings.copy().astype(np.float32)
        self.prev_portfolio_value = float(self.cash + np.sum(self.holdings * self.prices[0]))
        self.total_step = 0
        self.last_action = np.zeros(self.N, dtype=np.int32)

        self.date_memory = []
        self.asset_memory = []
        self.actions_memory = []

        # rolling windows and previous weights for turnover
        self.ret_window = deque(maxlen=60)
        self.pv_window = deque(maxlen=60)
        self.weights_prev = np.zeros(self.N)
        self.pv_window.append(self.prev_portfolio_value)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.reset_internal_store()
        return self._get_obs(), {}

    def _get_obs(self):
        price_today = self.prices[self.day]
        tech_today = self.tech[self.day].reshape(-1)
        turb = np.array([self.turbulence[self.day]], dtype=np.float32)

        obs = np.concatenate([
            np.array([self.cash], dtype=np.float32),
            self.holdings.astype(np.float32),
            price_today.astype(np.float32),
            tech_today.astype(np.float32),
            turb
        ]).astype(np.float32)

        return obs

    def _normalize_turbulence(self, turb_value: float) -> float:
        if self.turbulence_normalization == "max":
            return float(turb_value) / (self._turb_max + 1e-12)
        if self.turbulence_normalization == "zscore":
            return (float(turb_value) - self._turb_mean) / (self._turb_std + 1e-12)
        
        return float(turb_value)

    def _compute_reward(self, portfolio_value):
        # Instant log return
        V_t = self.prev_portfolio_value
        V_t1 = portfolio_value
        instant_ret = np.log((V_t1 + 1e-12) / (V_t + 1e-12))

        # update rolling windows
        self.ret_window.append(instant_ret)
        self.pv_window.append(portfolio_value)

        # Rolling volatility (20)
        if len(self.ret_window) >= 20:
            rolling_vol = np.std(list(self.ret_window)[-20:])
        else:
            rolling_vol = 0.0

        # Rolling max drawdown (60)
        pv_arr = np.array(self.pv_window)
        if len(pv_arr) >= 2:
            peak = np.maximum.accumulate(pv_arr)
            drawdowns = (peak - pv_arr) / (peak + 1e-12)
            rolling_dd = float(np.max(drawdowns))
        else:
            rolling_dd = 0.0

        # Turnover penalty (weights difference)
        prices = self.prices[self.day]
        holdings_value = self.holdings * prices
        portfolio_v = float(holdings_value.sum() + self.cash)
        weights = holdings_value / (portfolio_v + 1e-12)
        turnover = np.sum(np.abs(weights - self.weights_prev))
        self.weights_prev = weights.copy()

        # Turbulence penalty
        turb_raw = float(self.turbulence[self.day])
        turb_norm = self._normalize_turbulence(turb_raw)
        turb_norm = float(np.clip(turb_norm, 0.0, 10.0))

        # Compose reward (coefficients are tunable env params)
        lambda_vol = 0.25
        lambda_dd = 0.5
        lambda_turn = 1e-4
        lambda_turb = float(self.turbulence_penalty_coef)

        raw_reward = (
            instant_ret
            - lambda_vol * rolling_vol
            - lambda_dd * rolling_dd
            - lambda_turn * turnover
            - lambda_turb * turb_norm   # turbulence subtracts from reward
        )

        return float(raw_reward * self.reward_scaling)

    def step(self, action: np.ndarray):
        assert action.shape == (self.N,), f"action shape must be {(self.N,)}, got {action.shape}"

        prices = self.prices[self.day]

        # scale action to integer share deltas
        raw_shares = (np.clip(action, -1.0, 1.0) * float(self.hmax)).astype(np.float32)
        shares_delta = np.trunc(raw_shares).astype(np.int32)

        self.last_action = shares_delta.copy()

        # SELL
        sell_units = np.abs(np.minimum(shares_delta, 0)).astype(np.int32)
        if np.any(sell_units):
            actual_sell = np.minimum(sell_units, self.holdings.astype(np.int32))
            revenue = (actual_sell.astype(np.float32) * prices) * (1.0 - self.sell_cost_pct)
            self.cash += float(np.sum(revenue))
            self.holdings = self.holdings - actual_sell.astype(np.float32)

        # BUY (skip if nothing to buy)
        buy_units = shares_delta.clip(min=0).astype(np.int32)
        if np.any(buy_units):
            space_left = (self.max_stock_holding - self.holdings).clip(min=0).astype(np.int32)
            actual_buy = np.minimum(buy_units, space_left)
            cost_wo_fee = actual_buy.astype(np.float32) * prices
            cost_w_fee = cost_wo_fee * (1.0 + self.buy_cost_pct)
            total_cost = float(np.sum(cost_w_fee))

            if not self.allow_negative_cash and total_cost > self.cash and np.sum(cost_w_fee) > 0.0:
                scale = float(self.cash / total_cost)
                if scale < 1e-8:
                    actual_buy = np.zeros_like(actual_buy)
                    total_cost = 0.0
                else:
                    scaled = np.floor(actual_buy.astype(np.float32) * scale).astype(np.int32)
                    actual_buy = scaled
                    cost_wo_fee = actual_buy.astype(np.float32) * prices
                    cost_w_fee = cost_wo_fee * (1.0 + self.buy_cost_pct)
                    total_cost = float(np.sum(cost_w_fee))

            self.cash -= total_cost
            self.holdings = self.holdings + actual_buy.astype(np.float32)


        self.day += 1
        self.total_step += 1
        done = (self.day >= (self.T - 1))

        current_prices = self.prices[self.day]
        portfolio_value = float(self.cash + np.sum(self.holdings * current_prices))

        reward = self._compute_reward(portfolio_value)
        self.prev_portfolio_value = portfolio_value

        date_val = None if self.dates is None else self.dates[self.day]
        self.date_memory.append(date_val if date_val is not None else int(self.day))
        self.asset_memory.append(portfolio_value)
        self.actions_memory.append(self.last_action.copy())

        info = {
            "total_assets": portfolio_value,
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
            "day": int(self.day),
            "date": None if self.dates is None else self.dates[self.day],
            "action": self.last_action.copy(),
        }

        obs = self._get_obs()
        return obs, reward, done, False, info

    def get_sb_env(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self):
        df = pd.DataFrame(self.actions_memory)
        if hasattr(self, "stock_dim") and df.shape[1] == self.stock_dim:
            pass
        df.insert(0, "date", self.date_memory)
        return df
