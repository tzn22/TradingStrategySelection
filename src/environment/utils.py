import numpy as np
from finrl.agents.stablebaselines3 import models as finrl_models
from stable_baselines3.common.callbacks import CallbackList


def safe_on_rollout_end(self) -> bool:
    buffer = getattr(self.model, "rollout_buffer", None)
    if buffer is None:
        buffer = getattr(self.model, "replay_buffer", None)

    rewards = None
    if buffer is not None and hasattr(buffer, "rewards"):
        rewards = np.array(buffer.rewards).flatten()
    elif isinstance(self.locals, dict):
        rewards = self.locals.get("rewards") or self.locals.get("reward")
        if rewards is not None:
            rewards = np.array(rewards).flatten()

    if rewards is not None and rewards.size > 0:
        self.logger.record("train/reward_min", float(np.min(rewards)))
        self.logger.record("train/reward_mean", float(np.mean(rewards)))
        self.logger.record("train/reward_max", float(np.max(rewards)))
    else:
        self.logger.record("train/reward_min", None)
        self.logger.record("train/reward_mean", None)
        self.logger.record("train/reward_max", None)
    return True

# finrl_models.TensorboardCallback._on_rollout_end = safe_on_rollout_end

def safe_train_model(model, tb_log_name, total_timesteps=50000, callbacks=None):
    base = finrl_models.TensorboardCallback()
    extra = []
    if callbacks:
        extra = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
    combo = CallbackList([base, *extra]) if extra else base
    return model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=combo)

# finrl_models.DRLAgent.train_model = staticmethod(safe_train_model)