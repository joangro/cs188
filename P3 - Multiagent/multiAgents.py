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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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
             
        "*** YOUR CODE HERE ***"
                
        # Always go to the next state if there is a reward
        if successorGameState.isWin():
            return 10000
        
        # If there is a ghost on the next state, avoid being there
        if successorGameState.isLose(): 
            return -10000
        
        # To avoid letting the pacman stop 
        if action == 'Stop':
            return -10000
           
        # 1. Look position to closest reward
        foodPos = newFood.asList()
        closestFoodDist = 10000
        for i in range(len(foodPos)):
          newDist = manhattanDistance(foodPos[i],newPos) # calculate distance using Manhattan Dist.
          if newDist < closestFoodDist:
            closestFoodDist = newDist
        
        # create new measurement, making the state farthest away to the closest food the least desirable (less score)
        newScore = -closestFoodDist

        # 2. Calculate dist from pacman to ghost, now using successor states
        for i in range(len(newGhostStates)):
          ghostPos = successorGameState.getGhostPosition(i+1)
          # If there is a ghost next to the successor state, avoid going there
          if manhattanDistance(newPos,ghostPos) < 2:
            newScore -= 10000
        # return sum of the scores
        return newScore + successorGameState.getScore()


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

        def minimize(gameState, depth, agentcounter):
            minState = ["", 10000] # save action and value 
            # for every ghost action 
            for action in gameState.getLegalActions(agentcounter):
                currentState = gameState.generateSuccessor(agentcounter, action) # get next state
                # we keep calling the minMax function to get to the end of the branches 
                current = minMax(currentState, depth, agentcounter + 1) 
                # we only keep the score 
                if type(current) is not list:
                    newValue = current
                else:
                    newValue = current[1]
                if newValue < minState[1]:
                    minState = [action, newValue]
            return minState
        
        # same as the minimize function 
        def maximize(gameState, depth, agentcounter):
            maxState = ["", -10000] # save action and value 
            
            for action in gameState.getLegalActions(agentcounter):
                currentState = gameState.generateSuccessor(agentcounter, action)
                current = minMax(currentState, depth, agentcounter + 1)
                if type(current) is not list:
                    newValue = current
                else:
                    newValue = current[1]
                if newValue > maxState[1]:
                    maxState = [action, newValue]
            return maxState

        # pseudocode taken from en.wikipedia.org/wiki/Minimax
        # and help from oleksiirenov.blogspot.com.es/2015/03/designing-agents-algorithms-for-pacman.html
        # this first function chooses if the current node needs to be evaluated by the minimizer agent or the maximizer agent, depending on the depth.
        # It's the first function called to initialitzate and it's called by the agents
        def minMax(gameState, depth, agentcounter):
            
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0
            
            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return maximize(gameState, depth, agentcounter)
            else:
                return minimize(gameState, depth, agentcounter)
            
        actionsList = minMax(gameState, 0, 0)
        return actionsList[0]





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # We implement alpha and beta in the same algorithm as before
        def minimize(gameState, depth, agentcounter, a, b):
            minState = ["", 10000]

            for action in gameState.getLegalActions(agentcounter):
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minMax(currState, depth, agentcounter + 1, a, b)

                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]
                if newVal < minState[1]:
                    minState = [action, newVal]
                # now, if the new value is smaller than the alpha, we return the value without calculating any more values on the node
                if newVal < a:
                    return [action, newVal]
                 # in any other case, update the alpha with the minimum value
                b = min(b, newVal)
            return minState

        def maximize(gameState, depth, agentcounter, a, b):
            maxState = ["", -10000]
            actions = gameState.getLegalActions(agentcounter)

            for action in gameState.getLegalActions(agentcounter):
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minMax(currState, depth, agentcounter + 1, a, b)

                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]

                if newVal > maxState[1]:
                    maxState = [action, newVal]
                # if the value is bigger than the beta, we don't calculate more values
                if newVal > b:
                    return [action, newVal]
                # in any other case, update the alpha with the maximum value
                a = max(a, newVal)
            return maxState

        def minMax(gameState, depth, agentcounter, a, b):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return maximize(gameState, depth, agentcounter, a, b)
            else:
                return minimize(gameState, depth, agentcounter, a, b)

        actionsList = minMax(gameState, 0, 0, -10000, 10000)
        return actionsList[0]



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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

