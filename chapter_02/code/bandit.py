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
        # Only update if at least one action has been performed
        if self.action_counter[action] > 0:
            # Compute reward of the action
            self.last_reward = self.reward(action)
            # Update alpha
            alpha = self.update_learning_rate(action)
            # Update average reward for the action
            self.avg_rewards[action] += alpha * (
                self.last_reward - self.avg_rewards[action]
            )
        return self

    def choose_action(self):
        """Choose next action (greedy-eps)"""
        if np.random.rand() >= self.eps:
            # Go greedy
            return np.argmax(self.avg_rewards)
        else:
            # Return random action
            return np.random.choice(self.k)

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
    """General class for a bandit problem"""

    """Supports stationary or non-stationary problems"""

    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary
        self.reset()

    def reset(self):
        # Scalars
        self.steps = 0
        self.last_reward = 0

        # Counters and avg rewards
        self.action_counter = np.zeros(self.k, dtype=np.int32)
        self.avg_rewards = self.set_initial_rewards(self.k)

        # Action Values and Optimal Action
        self.action_values = self.set_action_values(self.k)
        self.optimal_action = np.argmax(self.action_values)

        return self

    def set_action_values(self, k):
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


spec_sa = [("eps", float64), ("init_action_val", float64)]


@jitclass(spec + spec_sa)
class SampleAverage(MultiArmedBandit, BanditMixin):
    """Class for sample average action-value method"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__

    def __init__(self, k, eps=0.1, init_action_val=0, stationary=True):
        self.eps = eps
        self.init_action_val = init_action_val

        self.__init__MultiArmedBandit(k, stationary)

    def set_initial_rewards(self, k):
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def update_learning_rate(self, action):
        # Learning rate is given by 1/n where n is the number a certain action has been
        #  already selected
        return 1 / self.action_counter[action]


spec_cs = [
    ("eps", float64),
    ("init_action_val", float64),
    ("alpha", float64),
    ("o", float64),
]


@jitclass(spec + spec_cs)
class ConstantStepSize(MultiArmedBandit, BanditMixin):
    """Class for sample average with constant step-size learning rate"""

    __init__MultiArmedBandit = MultiArmedBandit.__init__

    def __init__(self, k, eps=0.1, init_action_val=0, alpha=0.1, stationary=True):
        self.eps = eps
        self.init_action_val = init_action_val
        self.alpha = alpha
        self.o = 0

        self.__init__MultiArmedBandit(k, stationary)

    def set_initial_rewards(self, k):
        # Initial rewards are `init_action_val`
        return self.init_action_val * np.ones(self.k, dtype=np.float64)

    def update_learning_rate(self, action):
        # Use unbiased trick to update alpha for constant step-size learning
        self.o = self.o + self.alpha * (1 - self.o)
        return self.alpha / self.o
