# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        self.values = util.Counter()  # Initialize with the dict of 0
        self.runValueIteration()  # Run the value iteration

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # For each interaction we calculate the best action for all states
        for i in range(self.iterations):

            updatedValues = util.Counter()

            for state in states:
                # We get the best action defined by the function we just did computeActionFromValues
                action = self.getAction(state)

                if action != None:
                    # If its not terminal, we get the best Q-value for that state
                    updatedValues[state] = self.getQValue(state, action)

            self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        # we get the next state and probabilities
        nextStateandProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextStateandProb in nextStateandProbs:
            # We break into 2 variables = s' and T(s, a, s')
            nextState, probabilities = nextStateandProb

            # Calculate reward for the transition, r(s, a, s')
            reward = self.mdp.getReward(state, action, nextState)
            # We get the V_k(s')
            value = self.getValue(nextState)

            # Calculate sum(T() * [r(s,a,s') + discount * V_k(s')])
            total += probabilities * (reward + self.discount * value)

        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # We get all the legal actions
        legalActions = self.mdp.getPossibleActions(state)

        # If theres no legal actions we return none
        if len(legalActions) == 0:
            return None
        else:
            # initialize a qvalues for holding the Qvalues
            qValues = util.Counter()

            # For each action, we get the QValues
            for action in legalActions:
                qValues[action] = self.getQValue(state, action)

            # Return the action with the highest QValue
            return qValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        self.values = util.Counter()  # Initialize with the dict of 0
        self.runValueIteration()  # Run the value iteration

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        state_size = len(states)
        aux = 0
        # For each interaction we calculate the best action for just the actual state (cyclic)
        for i in range(self.iterations):

            # Aux variable to not go beyond the index of the states, when reach the final state, start again from 0
            if (aux == state_size):
                aux = 0

            # We get the best action defined by the function we just did computeActionFromValues
            action = self.getAction(states[aux])
            updatedValues = 0

            if action != None:
                # If its not terminal, we get the best Q-value for that state
                updatedValues = self.getQValue(
                    states[aux], action)

            # We update the value just for this state
            self.values[states[aux]] = updatedValues

            # Increment auxiliar variable
            aux += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.iterations = iterations
        self.discount = discount
        self.mdp = mdp
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # We get the states
        states = self.mdp.getStates()
        oldValues = self.values

        # Initialize predecessors
        predecessors = {}

        # initialize with a set()
        for state in states:
            predecessors[state] = set()

        # Create a priority queue
        priorityQueue = util.PriorityQueue()

        # iterate over states
        for state in states:
            # Get possible actions
            actions = self.mdp.getPossibleActions(state)

            # Initialize a new counter of qvalues
            values = util.Counter()

            # We will need also the value for the current state
            currentValue = oldValues[state]

            # Loop through all the actions
            for action in actions:
                # Get the transition and probs
                trans = self.mdp.getTransitionStatesAndProbs(
                    state, action)

                # If the prob of reaching this state is not 0, we append those states to the predecessors
                for nextState, prob in trans:
                    if prob != 0.0:
                        predecessors[nextState].add(state)

                # Get the value for this action
                values[action] = self.computeQValueFromValues(state, action)

                # Calculate the diff as: current value - QMaxValue
                diff = abs(currentValue - values[values.argMax()])

                # Insert that state into the priority Queue
                priorityQueue.update(state, -diff)

        # Now we loop through all the interations
        for i in range(self.iterations):

            # Check if the priority queue is empty, if not, continue
            if priorityQueue.isEmpty():
                return

            # Pop a state from the queue
            state = priorityQueue.pop()

            # If not a terminal state, we calculate the best QValue
            if not self.mdp.isTerminal(state):
                QValues = util.Counter()

                for action in self.mdp.getPossibleActions(state):
                    QValues[action] = self.computeQValueFromValues(
                        state, action)

                oldValues[state] = QValues[QValues.argMax()]

            # We iterate over the predecessors of that state
            for p in predecessors[state]:
                QValues_p = util.Counter()
                # Get the possible actions
                actions = self.mdp.getPossibleActions(p)

                # Loop through the actions to find the best action with the highest QValue
                for action in actions:
                    QValues_p[action] = self.computeQValueFromValues(p, action)

                # We calculate the diff with the current value for this state and the best possible value
                # ABS functions is for returning absolute values (non negative)
                diff = abs(oldValues[p] - QValues_p[QValues_p.argMax()])

                # If the differente is greater than theta, we update the priority of this state in the priority queue
                if (diff > self.theta):
                    priorityQueue.update(p, -diff)
