import numpy as np
from numpy.random import default_rng

rng = default_rng()


def random_walk(t: int, k: int, sigma: float = 0.01):
    """Perform k random walks of t time steps"""
    """Assume the initial position to be x=0"""

    # Perform random walk (cumulative sum of random steps)
    return np.cumsum(rng.normal(0, sigma, size=(t, k)), axis=0)


class BanditMixin:
    """General purpose methods for a Multi Armed Bandit"""

    def reward(self, action):
        """Perform Action"""
        # The reward for action k is drawn from a normal distribution of mean q(k)
        #  and variance 1
        return rng.normal(self.action_values[action], 1)

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
        if rng.uniform() >= self.eps:
            # Go greedy
            return np.argmax(self.avg_rewards)
        else:
            # Return random action
            return np.random.choice(self.k)

    def train(self, steps):
        """Train the agent"""
        output = np.zeros(shape=(steps, 3))
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
        self.action_counter = np.zeros(self.k)
        self.avg_rewards = self.set_initial_rewards(self.k)

        # Action Values and Optimal Action
        self.action_values = self.set_action_values(self.k)
        self.optimal_action = np.argmax(self.action_values)

        return self

    def set_action_values(self, k):
        """Set initial action values"""
        if self.stationary:
            # Return a normal vector of size k
            return rng.standard_normal(self.k)
        else:
            # Return a unitary vector of size k
            return np.ones(shape=(self.k,))

    def test_action_values(self, t):
        """Test trajectory of action values for a stationary problem"""
        if self.stationary:
            # Return a normally distributed matrix of size k and length t
            return rng.normal(loc=self.action_values, size=(t, self.k))
        else:
            # Perform a random walk with initial values being the action values
            return self.action_values + random_walk(t, self.k)

    def update_action_values(self):
        """For a stationary problem, reward probabilities do not change"""
        """For a non-stationary problem, reward probabilities perform a random walk"""
        if not self.stationary:
            # Perform the random walk by adding a normally distributed vector of size k
            #  to the action values
            self.action_values += rng.normal(0, 0.01, size=(self.k,))
            self.optimal_action = np.argmax(self.action_values)
        return self


class SampleAverage(MultiArmedBandit, BanditMixin):
    """Class for sample average action-value method"""

    def __init__(self, k, eps=0.1, stationary=True):
        super().__init__(k, stationary)
        self.eps = eps

    def set_initial_rewards(self, k):
        # Initial rewards are 0
        return np.zeros(k)

    def update_learning_rate(self, action):
        # Learning rate is given by 1/n where n is the number a certain action has been
        #  already selected
        return 1 / self.action_counter[action]


class ConstantStepSize(SampleAverage):
    """Class for sample average with constant step-size learning rate"""

    def __init__(self, k, eps=0.1, stationary=True, alpha=0.1):
        super().__init__(k, eps, stationary)
        self.alpha = alpha
        self.o = 0

    def update_learning_rate(self, action):
        # Use unbiased trick to update alpha for constant step-size learning
        self.o = self.o + self.alpha * (1 - self.o)
        return self.alpha / self.o
