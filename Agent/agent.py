
class Agent(object):
    """Abstract class for an agent.
    """
    def __init__(self, epsilon=None):
        """Init.

        Args:
            epsilon is optional exploration starting rate
        """
        self.epsilon = epsilon

    def act(self, state):
        """Action function.

        This function takes a state (from an environment) as an argument and
        returns an action.

        Args:
            state (numpy.array): state vector

        Returns:
            np.array: numpy array of the action to take
        """
        raise NotImplementedError()

    def observe(self, state, action, reward, next_state, terminal, *args):
        """Observe function.

        This function takes a state, a reward and a terminal boolean and returns a loss value. This is only used for learning agents.

        Args:
            state (numpy.array): state vector
            action (numpy.array): action vector
            reward (float): reward value
            next_state (numpy.array): next state vector
            terminal (bool): whether the game is over or not

        Returns:
            float: value of the loss
        """
        raise NotImplementedError()

    def end(self):
        """End of episode logic.
        """
        pass