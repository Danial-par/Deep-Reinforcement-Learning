import os
import random
import numpy as np
import torch
import gym

# -------------------------------
# Environment Factory
# -------------------------------
def make_env(env_id, max_episode_steps=None):
    """
    Creates a gym environment with optional custom episode length.
    """
    def _init():
        env = gym.make(env_id)
        if max_episode_steps is not None:
            env._max_episode_steps = max_episode_steps
        return env
    return _init

# -------------------------------
# Seeding Utilities
# -------------------------------
def set_random_seeds(seed):
    """
    Ensures reproducibility by setting seeds for numpy, torch, and random.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------
# Logging Decorator
# -------------------------------
def mean_of_list(func):
    """
    Decorator that takes a list of lists and computes the mean for all except the last 4,
    then adds explained variance of the last 4 items (usually value predictions).
    """
    def wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [
            sum(lst) / len(lst) for lst in lists[:-4]
        ] + [
            explained_variance(lists[-4], lists[-3]),
            explained_variance(lists[-2], lists[-1])
        ]
    return wrapper

# -------------------------------
# Evaluation Metric
# -------------------------------
def explained_variance(ypred, y):
    """
    Computes 1 - Var[y - ypred] / Var[y]
    A value close to 1 indicates high predictive performance.
    """
    assert y.ndim == 1 and ypred.ndim == 1
    variance_y = np.var(y)
    if variance_y == 0:
        return np.nan
    return 1 - np.var(y - ypred) / variance_y

# -------------------------------
# Running Mean and Variance
# -------------------------------
class RunningMeanStd:
    """
    Maintains running mean and variance (for normalization).
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
