import numpy as np
import random
class ThompsonSampling:
    def __init__(self, n_arms, explore=10):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Success count (prior)
        self.beta = np.ones(n_arms)   # Failure count (prior)
        self.iteration = 0
        self.random_sampler = RandomSampling(n_arms, weights=[1/n_arms] * n_arms)  # Uniform prior
        self.explore = explore  

    def sample(self):
        if self.iteration < self.explore:
            return self.random_sampler.sample()
        sampled = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled)

    def update(self, arm, reward):
        self.iteration += 1
        if not isinstance(reward, (int, np.integer)) or reward not in (0, 1):
            reward = int(reward > 0)  # Convert to binary reward
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

class GaussianThompsonSampling:
    def __init__(self, n_models, prior_mean=0.5, prior_var=0.1, noise_var=0.05, explore=10):
        self.n_models = n_models
        self.mu = np.full(n_models, prior_mean, dtype=float)
        self.lmbda = np.full(n_models, 1.0 / prior_var, dtype=float)  # precision
        self.noise_var = noise_var
        self.iteration = 0
        self.random_sampler = RandomSampling(n_models, weights=[1/n_models] * n_models)  # Uniform prior
        self.explore = explore  # Number of iterations to explore before sampling from Gaussian

    def sample(self):
        if self.iteration < self.explore:
            return self.random_sampler.sample()
        sampled = np.random.normal(self.mu, np.sqrt(1.0 / self.lmbda))
        return np.argmax(sampled)

    def update(self, model_idx, reward):
        self.iteration += 1
        if not isinstance(reward, (int, float, np.number, tuple, list)) or not np.isfinite(reward):
            raise ValueError(f"Expected finite float reward, got {reward}")
        if isinstance(reward, (tuple, list)):
            reward = np.mean(reward)  # Use mean if reward is a list or tuple
        # Bayesian update for Gaussian with known noise variance
        self.lmbda[model_idx] += 1.0 / self.noise_var
        self.mu[model_idx] = (
            (self.lmbda[model_idx] - 1.0 / self.noise_var) * self.mu[model_idx] + reward / self.noise_var
        ) / self.lmbda[model_idx]

class RandomSampling:
    def __init__(self, n_models, weights, explore=10):
        self.n_models = n_models
        self.weights = weights
        self.random_state = random.Random()
        ## normalize weights to ensure they sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in weights]

    def sample(self):
        return self.random_state.choices(range(self.n_models), weights=self.weights, k=1)[0]

    def update(self, model_idx, reward):
        # No update needed for random sampling
        pass

SAMPLING_FUNCTIONS = {
    "thompson": ThompsonSampling,
    "gaussian_thompson": GaussianThompsonSampling,
    "random": RandomSampling,
}

def get_sampling_function(fn_name, n_models, **kwargs):
    if fn_name not in SAMPLING_FUNCTIONS:
        raise ValueError(f"Unknown sampling function: {fn_name}")    
    return SAMPLING_FUNCTIONS[fn_name](n_models, **kwargs)
