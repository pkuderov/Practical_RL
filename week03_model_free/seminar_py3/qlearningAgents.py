# qlearningAgents.py
# ------------------
# based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from collections import defaultdict
import numpy as np
from learningAgents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate aka gamma)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
        - self.getQValue(state,action)
          which returns Q(state,action)
        - self.setQValue(state,action,value)
          which sets Q(state,action) := value

      !!!Important!!!
      NOTE: please avoid using self._qValues directly to make code cleaner
    """

    def __init__(self, **args):
        "We initialize agent and Q-values here."
        ReinforcementAgent.__init__(self, **args)
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def setQValue(self, state, action, value):
        """
          Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

#---------------------#start of your code#---------------------#

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """

        possible_actions = self.getLegalActions(state)
        if not possible_actions:
            # terminate state, it's value = 0.
            return 0.

        action_values = self._qvalues[state]
        value = max(action_values[action] for action in possible_actions)
        return value

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.
        """
        possible_actions = self.getLegalActions(state)
        if not possible_actions:
            # terminate state, no actions
            return None

        best_action, best_value = None, None
        action_values = self._qvalues[state]
        for action in possible_actions:
            value = action_values[action]
            if best_action is None or value > best_value:
                best_action, best_value = action, value

        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state, including exploration.  

          With probability self.epsilon, we should take a random action.
          otherwise - the best policy action (self.getPolicy).
        """
        possible_actions = self.getLegalActions(state)
        if not possible_actions:
            return None

        # agent parameters:
        epsilon = self.epsilon
        if np.random.uniform(low=0., high=1.) <= epsilon:
            action = np.random.choice(possible_actions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          You should do your Q-Value update here.
        """
        # agent parameters
        r, gamma, alpha = reward, self.discount, self.alpha
        Qsa = self._qvalues[state][action]
        Vns = self.getValue(nextState)

        new_qvalue = (1 - alpha) * Qsa + alpha * (r + gamma * Vns)
        self.setQValue(state, action, new_qvalue)

#---------------------#end of your code#---------------------#


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    pass
