import numpy as np
from numba import int32, float64, boolean, njit
from numba.experimental import jitclass


@njit
def random_walk(t: int, k: int, sigma: float = 0.01):
    """Perform k random walks of t time steps"""
    """Assume the initial position to be x=0"""

    # Perform random walk (cumulative sum of random steps)
    random_matrix = np.zeros(shape=(t, k), dtype=np.float64)
    for i in range(k):
        random_matrix[:, i] = np.cumsum(sigma * np.random.randn(t))
    return random_matrix


class BanditMixin:
    """General purpose methods for a Multi Armed Bandit"""

    def reward(self, action):
        """Perform Action"""
        # The reward for action k is drawn from a normal distribution of mean q(k)
        #  and variance 1
        return self.action_values[action] + np.random.randn()

    def update_reward(self, action):
        """Update sample average reward"""
        # Compute reward of the action
        self.last_reward = self.reward(action)
        # Update alpha
        alpha = self.update_learning_rate(action)
        # Update average reward for the action
        self.avg_rewards[action] += alpha * (
            self.last_reward - self.avg_rewards[action]
        )
        return self

    def train(self, steps):
        """Train the agent"""
        output = np.zeros(shape=(steps, 3), dtype=np.float64)
        for step in range(steps):
            self.steps += 1
            # Choose an action
            action = self.choose_action()
            # Update counter and reward
            self.action_counter[action] += 1
            self.update_reward(action)
            # Write output
            output[step, 0] = action
            output[step, 1] = self.last_reward
            output[step, 2] = self.optimal_action
            # Update action values
            self.update_action_values()

        # Split the 3 outputs for convenience
        return output[:, 0], output[:, 1], output[:, 2]


spec = [
    ("k", int32),
    ("stationary", boolean),
    ("steps", int32),
    ("last_reward", float64),
    ("action_counter", int32[:]),
    ("avg_rewards", float64[:]),
    ("action_values", float64[:]),
    ("optimal_action", int32),
]


class MultiArmedBandit:
    """Abstract class for a bandit problem"""

    """Supports stationary or non-stationary problems"""

    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary
        self.reset()

    def reset(self):
        """Define Bandit properties"""
        # Scalars
        self.steps = 0
        self.last_reward = 0

        # Counters and avg rewards
        self.action_counter = np.zeros(self.k, dtype=np.int32)
        self.avg_rewards = self.set_initial_rewards()

        # Action Values and Optimal Action
        self.action_values = self.set_action_values()
        self.optimal_action = np.argmax(self.action_values)

        return self

    def set_action_values(self):
        """Set initial action values"""
        if self.stationary:
            # Return a normal vector of size k
            return np.random.randn(self.k)
        else:
            # Return a unitary vector of size k
            return np.ones(shape=(self.k,), dtype=np.float64)

    def test_action_values(self, t):
        """Test trajectory of action values for a stationary problem"""
        if self.stationary:
            # Return a normally distributed matrix with mean q(k) and variance 1
            # of size k and length t
            return self.action_values + np.random.randn(t, self.k)
        else:
            # Perform a random walk with initial values being the action values
            return self.action_values + random_walk(t, self.k)

    def update_action_values(self):
        """For a stationary problem, reward probabilities do not change"""
        """For a non-stationary problem, reward probabilities perform a random walk"""
        if not self.stationary:
            # Perform the random walk by adding a normally distributed vector of size k
            #  to the action values
            self.action_values += 0.01 * np.random.randn(self.k)
            self.optimal_action = np.argmax(self.action_values)
        return self


# Greedy-Epsilon Family
class GreedyEpsMixin(BanditMixin):
    """General purpose methods for a GreedyEps method"""

    def set_initial_rewards(self):
        """Set the initial rewards"""
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def choose_action(self):
        """Choose next action (greedy-eps)"""
        if np.random.rand() >= self.eps:
            # Go greedy
            return np.argmax(self.avg_rewards)
        else:
            # Return random action
            return np.random.choice(self.k)


spec_sa = [("eps", float64), ("init_action_val", float64)]


@jitclass(spec + spec_sa)
class GreedyEpsAgent(MultiArmedBandit, GreedyEpsMixin):
    """Class for sample average action-value method,
    with eps-greedy action selection"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__

    def __init__(self, k, eps=0.1, init_action_val=0, stationary=True):
        self.eps = eps
        self.init_action_val = init_action_val
        self.__init__MultiArmedBandit(k, stationary)

    def update_learning_rate(self, action):
        """Update learning rate for action `action`. Learning rate is given by 1/n
        where n is the number a certain action has been already selected"""
        return 1 / self.action_counter[action]


spec_cs = [
    ("eps", float64),
    ("init_action_val", float64),
    ("alpha", float64),
    ("o", float64),
]


@jitclass(spec + spec_cs)
class ConstantStepSizeGreedyEpsAgent(MultiArmedBandit, GreedyEpsMixin):
    """Class for sample average with constant step-size learning rate,
    with eps-greedy action selection"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset

    def __init__(self, k, eps=0.1, init_action_val=0, alpha=0.1, stationary=True):
        self.eps = eps
        self.init_action_val = init_action_val
        self.alpha = alpha
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.o = 0

    def update_learning_rate(self, action):
        """Update learning rate for action `action`. Use unbiased trick to update alpha
        for constant step-size learning"""
        self.o = self.o + self.alpha * (1 - self.o)
        return self.alpha / self.o


# UCB family
class UCBMixin(BanditMixin):
    """General purpose methods for a UCB method"""

    def set_initial_rewards(self):
        """Set the initial rewards"""
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def choose_action(self):
        """Choose next action (UCB)"""
        return np.argmax(
            self.avg_rewards
            + self.c * np.sqrt(np.log(self.steps) / self.action_counter)
        )


spec_ucb = [
    ("c", float64),
    ("init_action_val", float64),
]


@jitclass(spec + spec_ucb)
class UCBAgent(MultiArmedBandit, UCBMixin):
    """Class for sample average action-value method,
    with Upper-Confidence-Bound action selection"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__

    def __init__(self, k, c=1, init_action_val=0, stationary=True):
        self.c = c
        self.init_action_val = init_action_val
        self.__init__MultiArmedBandit(k, stationary)

    def update_learning_rate(self, action):
        """Update learning rate for action `action`. Learning rate is given by 1/n
        where n is the number a certain action has been already selected"""
        return 1 / self.action_counter[action]


spec_wucb = [
    ("weight", float64),
    ("c", float64),
    ("init_action_val", float64),
    ("weighted_counter", float64[:]),
]


@jitclass(spec + spec_wucb)
class WeightedUCBAgent(MultiArmedBandit, UCBMixin):
    """Class for sample average action-value method with discount,
    with Upper-Confidence-Bound action selection"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset
    update_reward__UCBMixin = UCBMixin.update_reward

    def __init__(self, k, c=1, weight=0.999, init_action_val=0, stationary=True):
        self.c = c
        self.weight = weight
        self.init_action_val = init_action_val
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.weighted_counter = np.zeros(self.k, dtype=np.float64)

    def choose_action(self):
        """Choose next action (UCB)"""
        return np.argmax(
            self.avg_rewards
            + 2
            * self.c
            * np.sqrt(np.log(self.weighted_counter.sum()) / self.weighted_counter)
        )

    def update_reward(self, action):
        """Update sample average reward"""
        # Discount all weighted weighted counters
        self.weighted_counter *= self.weight
        # Perform normal update of the estimated reward
        self.update_reward__UCBMixin(action)
        return self

    def update_learning_rate(self, action):
        """Update learning rate for action `action`.
        Add 1 to the weighted counter if action is selected.
        Return the weighted counter"""
        self.weighted_counter[action] += 1
        return 1 / self.weighted_counter[action]


# Gradient Bandit family
class GradientBanditMixin(BanditMixin):
    """General purpose methods for a GradientBandit method"""

    def set_initial_rewards(self):
        """Set the initial rewards"""
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def compute_probabilities(self):
        """Compute action probabilities"""
        return self.action_preferences / self.action_preferences.sum()

    def choose_action(self):
        """Choose next action (Gradient Bandit)"""
        self.probabilities = self.compute_probabilities()
        return np.argmax(np.random.multinomial(1, self.probabilities))

    def update_reward(self, action):
        """Update sample average reward (for Bandit Algorithm)"""
        # Compute reward of the action
        self.last_reward = self.reward(action)
        # Update alpha
        alpha = self.update_learning_rate()
        # Update average reward for the action
        self.avg_rewards += alpha * (self.last_reward - self.avg_rewards)
        # Update the action preferences
        self.update_action_preferences(action)
        return self

    def update_action_preferences(self, action):
        """Update action preferences"""
        # General update
        update = (
            -self.alpha * (self.last_reward - self.avg_rewards) * self.probabilities
        )
        # Specific update for action `action` after selection
        update[action] += self.alpha * (self.last_reward - self.avg_rewards[action])
        self.action_preferences *= np.exp(update)
        return self


spec_gba = [
    ("alpha", float64),
    ("init_action_val", float64),
    ("action_preferences", float64[:]),
    ("probabilities", float64[:]),
]


@jitclass(spec + spec_gba)
class GradientBanditAgent(MultiArmedBandit, GradientBanditMixin):
    """Class for Gradient Bandit method"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset

    def __init__(self, k, alpha=0.1, init_action_val=0, stationary=True):
        self.alpha = alpha
        self.init_action_val = init_action_val
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.action_preferences = np.ones(self.k, dtype=np.float64)
        self.probabilities = self.compute_probabilities()
        return self

    def update_learning_rate(self):
        """Update learning rate."""
        # Use incremental update if problem is stationary
        return 1 / self.steps


spec_wgba = [
    ("alpha", float64),
    ("weight", float64),
    ("o", float64),
    ("init_action_val", float64),
    ("action_preferences", float64[:]),
    ("probabilities", float64[:]),
]


@jitclass(spec + spec_wgba)
class WeightedGradientBanditAgent(MultiArmedBandit, GradientBanditMixin):
    """Class for Gradient Bandit method, with weighted estimate reward"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset

    def __init__(self, k, alpha=0.1, weight=0.1, init_action_val=0, stationary=True):
        self.alpha = alpha
        self.weight = weight
        self.init_action_val = init_action_val
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.o = 0
        self.action_preferences = np.ones(self.k, dtype=np.float64)
        self.probabilities = self.compute_probabilities()

        return self

    def update_learning_rate(self):
        """Update learning rate for action `action`. Use unbiased trick to update weight
        for constant step-size learning"""
        self.o = self.o + self.weight * (1 - self.o)
        return self.weight / self.o


# Thompson Sampling family
class ThompsonSamplingMixin(BanditMixin):
    """General purpose methods for a Thompson Sampling method"""

    update_reward__BanditMixin = BanditMixin.update_reward

    def set_initial_rewards(self):
        """Set the initial rewards"""
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def update_reward(self, action):
        """Update sample average reward"""
        self.update_reward__BanditMixin(action)
        self.update_posterior(action)

    def choose_action(self):
        """Choose next action (Thomson Bandit)"""
        """Sample a value from the the posterior normal distribution.
         Return the max index"""
        return np.argmax((np.random.randn() / np.sqrt(self.tau)) + self.mu)


spec_th = [
    ("init_action_val", float64),
    ("mu", float64[:]),
    ("tau", float64[:]),
    ("tau_v", float64),
]


@jitclass(spec + spec_th)
class ThompsonSamplingAgent(MultiArmedBandit, ThompsonSamplingMixin):
    """Class for Thomson Sampling method -- constant variance"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset

    def __init__(self, k, tau_v=1, init_action_val=0, stationary=True):
        self.init_action_val = init_action_val
        self.tau_v = tau_v
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.mu = np.zeros(self.k, dtype=np.float64)
        self.tau = 0.0001 * np.ones(self.k, dtype=np.float64)
        return self

    def update_posterior(self, action):
        """Update posterior distribution"""
        mu = self.mu[action]
        tau = self.tau[action]

        self.tau[action] += self.tau_v
        self.mu[action] = ((mu * tau) + self.last_reward * self.tau_v) / self.tau[
            action
        ]

        return self

    def update_learning_rate(self, action):
        """Update learning rate for action `action`. Learning rate is given by 1/n
        where n is the number a certain action has been already selected"""
        return 1 / self.action_counter[action]


spec_wth = [
    ("init_action_val", float64),
    ("alpha", float64),
    ("o", float64),
    ("mu", float64[:]),
    ("tau", float64[:]),
    ("tau_v", float64),
    ("gamma", float64),
    ("mu_t", float64),
    ("tau_t", float64),
]


@jitclass(spec + spec_wth)
class WeightedThompsonSamplingAgent(MultiArmedBandit, ThompsonSamplingMixin):
    """Class for Thomson Sampling method with discount"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__
    reset__MultiArmedBandit = MultiArmedBandit.reset

    def __init__(self, k, gamma=0.01, tau_v=1, init_action_val=0, stationary=True):
        self.init_action_val = init_action_val
        self.tau_v = tau_v
        self.gamma = gamma
        self.alpha = 0.1
        self.__init__MultiArmedBandit(k, stationary)

    def reset(self):
        """Define Bandit properties"""
        self.reset__MultiArmedBandit()
        self.o = 0
        self.mu = np.zeros(self.k, dtype=np.float64)
        self.tau = 0.0001 * np.ones(self.k, dtype=np.float64)
        self.mu_t = 0
        self.tau_t = 0.0001
        return self

    def update_posterior(self, action):
        """Update posterior distribution"""
        # Create copy of prior
        mu = self.mu.copy()
        tau = self.tau.copy()

        # General and action-specific update of tau
        self.tau = (1 - self.gamma) * tau + self.gamma * self.tau_t
        self.tau[action] += self.tau_v

        # General and action-specific update of mu
        self.mu = (
            (1 - self.gamma) * (mu * tau) + self.gamma * (self.mu_t * self.tau_t)
        ) / self.tau
        self.mu[action] += (self.last_reward * self.tau_v) / self.tau[action]

        return self

    def update_learning_rate(self, action):
        """Update learning rate for action `action`. Use unbiased trick to update alpha
        for constant step-size learning"""
        self.o = self.o + self.alpha * (1 - self.o)
        return self.alpha / self.o
