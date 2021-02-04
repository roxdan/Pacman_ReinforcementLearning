# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math

# Question 6: Q-Learning


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # If this state has never been seen before we return 0.0
        if (state, action) not in self.values:
            self.values[(state, action)] = 0.0

        # Otherwise, we return the actual value for this state
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get actions list
        legalActions = self.getLegalActions(state)

        # If null, return 0
        if len(legalActions) == 0:
            return 0.0

        # Initialize a aux placeholder as a empty dict
        tmp = util.Counter()

        # Loop through every action and calculate the QValue
        for action in legalActions:
            tmp[action] = self.getQValue(state, action)

        # Return the maximum value of the best action
        return tmp[tmp.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get list of legal actions
        legalActions = self.getLegalActions(state)

        # If there are none, return None
        if len(legalActions) == 0:
            return None

        # Initialize a auxiliar placehold as a empty dict
        tmp = util.Counter()

        # Loop through the action and get the correspondent QValue
        for action in legalActions:
            tmp[action] = self.getQValue(state, action)

        # Return the action with the highest QValue
        return tmp.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        # If there are no legal actions, just return
        if len(legalActions) == 0:
            return action

        # Flip a coin to see if we get a random action, otherwise a optimal action
        if(util.flipCoin(self.epsilon)):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # We store the actual QValue
        oldQValue = self.values[(state, action)]

        # Calculate the sample as: r + y * maxQValue
        sample = reward + \
            (self.discount*self.computeValueFromQValues(nextState))

        # We update the values for this state and action based on the equation: (1 - alpha) * oldQValue + alpha * sample
        self.values[(state, action)] = (1 - self.alpha) * \
            oldQValue + (self.alpha) * sample

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


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

# Question 10: Approximate Q-Learning


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # We get the features for this state and action and also the weigths
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()

        # For every feature, we calculate it with the weights for this feature and sum it up
        qVal = 0
        for f in features:
            qVal += features[f] * weights[f]

        # Q(s, a) = sum (features*weight)
        # Return the sum of the QValues
        return qVal

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # We collect the features and the discount
        features = self.featExtractor.getFeatures(state, action)
        discount = self.discount

        # We calculate the difference equation: difference = (r + y * maxQ(nextState)) - QValue(state, action)
        difference = (reward + (discount * self.getValue(nextState))
                      ) - self.getQValue(state, action)

        # We iterate over the features and update the weights
        # Wi = Wi + alpha * difference * feature
        for f in features:
            self.weights[f] = self.weights[f] + \
                (self.alpha * difference * features[f])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # Printing some variables, just for debugging
            print('Weights: ', self.getWeights())
            print('Discount: ', self.discount)
            pass
