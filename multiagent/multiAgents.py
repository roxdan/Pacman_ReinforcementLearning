# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        print(legalMoves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        print(legalMoves[chosenIndex])
        #raw_input("Continue?")
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print("Action:", action)
        print("New pos:",newPos)
        #print(newFood)
        #print("New ghost states:",newGhostStates)
        #print("New scared times:",newScaredTimes)

        totalScore = 0.0
        oldFood = currentGameState.getFood()

        # Function to guide pacman in positions closer to food
        for x in xrange(oldFood.width):
          for y in xrange(oldFood.height):
            if oldFood[x][y]:
              d = manhattanDistance((x,y), newPos)

              if d==0:
                totalScore += 100
              else:
                totalScore += 1.0/(d*d)

        #Function to calculate distance between ghost
        for ghost in newGhostStates:
          d = manhattanDistance(ghost.getPosition(), newPos)
          if d<=1:
            if (ghost.scaredTimer!=0):
              totalScore += 2000
            else:
              totalScore -=200
        
        #Function to get the capsule
        for capsule in currentGameState.getCapsules():
          d = manhattanDistance(capsule, newPos)
          if d==0:
            totalScore +=1000
          else:
            totalScore += 1.0/(d)


        return totalScore
      
        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        import sys

        def result(gameState, agent, action):
          return gameState.generateSuccessor(agent, action)

        def utility(gameState):
          return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
          return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, agent, depth):
          if terminalTest(gameState, depth): return utility(gameState)
          v = -sys.maxint

          for a in gameState.getLegalActions(agent):
            v = max(v, min_value(result(gameState, agent, a), 1, depth))
            
          return v
        
        def min_value(gameState, agent, depth):
          if terminalTest(gameState, depth): return utility(gameState)
          v = sys.maxint

          for a in gameState.getLegalActions(agent):
            if (agent == gameState.getNumAgents()-1):
              v = min(v, max_value(result(gameState, agent, a), 0, depth-1))
            else:
              v = min(v, min_value(result(gameState, agent, a), agent+1, depth))

          return v

        v = -sys.maxint
        actions = []

        for a in gameState.getLegalActions(0):
          u = min_value(result(gameState, 0, a), 1, self.depth)

          if u == v: actions.append(a)
          elif u >= v: 
            v = u
            actions = [a]

        print ("Action value: ", v)
        return random.choice(actions)

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys

        def result(gameState, agent, action):
          return gameState.generateSuccessor(agent, action)

        def utility(gameState):
          return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
          return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, agent, depth, alpha, beta):
          if terminalTest(gameState, depth): return utility(gameState)
          v = -sys.maxint

          for a in gameState.getLegalActions(agent):
            v = max(v, min_value(result(gameState, agent, a), 1, depth, alpha, beta))
            
            if (v > beta): return v

            alpha = max(alpha, v)
          return v
        
        def min_value(gameState, agent, depth, alpha, beta):
          if terminalTest(gameState, depth): return utility(gameState)
          v = sys.maxint

          for a in gameState.getLegalActions(agent):
            if (agent == gameState.getNumAgents()-1):
              v = min(v, max_value(result(gameState, agent, a), 0, depth-1, alpha, beta))
            else:
              v = min(v, min_value(result(gameState, agent, a), agent+1, depth, alpha, beta))

            if (v < alpha): return v
            beta = min(beta, v)

          return v

        v = -sys.maxint
        actions = []
        alpha = -sys.maxint
        beta = sys.maxint

        for a in gameState.getLegalActions(0):
          u = min_value(result(gameState, 0, a), 1, self.depth, alpha, beta)

          if u == v: actions.append(a)
          elif u >= v: 
            v = u
            actions = [a]
          alpha = max(alpha, v)

        print ("Action value: ", v)
        return random.choice(actions)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        import sys

        def result(gameState, agent, action):
          return gameState.generateSuccessor(agent, action)

        def utility(gameState):
          #print("Utility: ", self.evaluationFunction(gameState))
          return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
          return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, agent, depth):
          if terminalTest(gameState, depth): return utility(gameState)
          v = -sys.maxint

          for a in gameState.getLegalActions(agent):
            v = max(v, min_value(result(gameState, agent, a), 1, depth))
            
          return v
        
        def min_value(gameState, agent, depth):
          if terminalTest(gameState, depth): return utility(gameState)
          v = []

          for a in gameState.getLegalActions(agent):
            if (agent == gameState.getNumAgents()-1):
              v.append(max_value(result(gameState, agent, a), 0, depth-1))
            else:
              v.append(min_value(result(gameState, agent, a), agent+1, depth))

          return sum(v)/float(len(v))

        v = -sys.maxint
        actions = []

        for a in gameState.getLegalActions(0):
          u = min_value(result(gameState, 0, a), 1, self.depth)

          if u == v: actions.append(a)
          elif u >= v: 
            v = u
            actions = [a]

            
        #print ("Action value: ", v)
        return random.choice(actions)
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Similar to my previous evaluation function, this function
      takes into account multiple features such as food distance, ghost distance and capsule distance
      The difference is that we now are evaluating states and not actions.
    """
    "*** YOUR CODE HERE ***"

    import sys

    # Get auxiliar values to evaluate
    actualFood = currentGameState.getFood()
    actualPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    # Get current score as a base for evaluation
    totalScore = scoreEvaluationFunction(currentGameState)

    # Function to guide pacman in positions closer to pellets (food)
    # If eat a pellet, receive a bonus, else receive the score proportional to its distance to the food
    for x in xrange(actualFood.width):
      for y in xrange(actualFood.height):
        if actualFood[x][y]:
          d = manhattanDistance((x,y), actualPos)

          if d==0:
            totalScore += 10000
          else:
            totalScore += 2.0/(d*d)


    #Function to calculate distance between ghost
    # If ghost is in scare timer, the score for eating it will be high
    for ghost in ghostStates:
      d = manhattanDistance(ghost.getPosition(), actualPos)
      if d<=1:
        if (ghost.scaredTimer!=0):
          totalScore += 8000
        else:
          totalScore -= 5000


    #Function to get the capsule
    # Its rewarding, but less than eating the pellets
    for capsule in currentGameState.getCapsules():
      d = manhattanDistance(capsule, actualPos)
      if d==0:
        totalScore += 9000
      else:
        totalScore += 2.0/(d)

    # This will make pacman lose points for every pellet remaining in the field
    # The idea is to make it want to eat the pellets
    totalScore += -400 * len(actualFood.asList())

    return totalScore

# Abbreviation
better = betterEvaluationFunction