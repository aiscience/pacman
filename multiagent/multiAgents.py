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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def lg(self, k,v):
        log=True
        if log:
            print("{k}={v}".format(k=k, v=v))

    def get_manhattan_distance(self, xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def get_euclidean_distance(self, xy1, xy2):
        return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

    def mean(self, ls):
        return sum(ls)/len(ls) if len(ls) > 0 else 0

    def scoreOnPosFoodGhost(self, position, foodGrid, ghostPos):
        # total_wall_cost = 1
        total_food_cost = 0
        # for w in wallList:
        #     total_wall_cost = total_wall_cost + self.get_euclidean_distance(w, position)
        foodList = foodGrid.asList()
        food_distances = [self.get_manhattan_distance(f, position) for f in foodList]
        food_distances.sort(reverse=True)
        factor = 1

        for f in food_distances:
            total_food_cost = total_food_cost + (f*factor)
            factor = factor * 10

        total_ghost_pos = 0.0
        min_ghost_dist = 9999
        for gp in ghostPos:
            ghost_dist = self.get_manhattan_distance(gp, position)
            min_ghost_dist = min(min_ghost_dist, ghost_dist)
            total_ghost_pos = total_ghost_pos + ghost_dist
        # total_ghost_pos = (total_ghost_pos * len(foodList) )**3
        self.lg("total_food_cost",total_food_cost)
        self.lg("total_ghost_pos",total_ghost_pos)
        if total_food_cost == 0:
            return 99999999999
        # return 10
        return (1/total_food_cost)*100 if min_ghost_dist > 1 else -1

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
        self.lg("currentGameState",currentGameState)
        self.lg("action",action)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        self.lg("successorGameState",successorGameState)
        newPos = successorGameState.getPacmanPosition()
        self.lg("newPos",newPos)
        newFood = successorGameState.getFood()
        self.lg("newFood",newFood)

        newGhostStates = successorGameState.getGhostStates()
        gs = [ghostState for ghostState in newGhostStates]
        self.lg("gs",str(gs))
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        self.lg("newScaredTimes",newScaredTimes)


        "*** YOUR CODE HERE ***"
        score = self.scoreOnPosFoodGhost(newPos, newFood, successorGameState.getGhostPositions())
        self.lg("score", score)
        return score

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
    def lg(self, k,v):
        log=True
        if log:
            print("{k}={v}".format(k=k, v=v))

    def max_val(self, gameState, depth, agentIndex):
        v = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            gs = gameState.generateSuccessor(agentIndex, action)
            depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)
            curr_val = self.val(gs, depth, newAgentIndex)
            v = max(v, curr_val)
        return v

        #
        # successor_states = [gameState.generateSuccessor(agentIndex, action) for action
        #                     in gameState.getLegalActions(agentIndex)]
        # return max([self.val(ss, depth, 1) for ss in successor_states])

    def min_val(self, gameState, depth, agentIndex):
        # self.lg('agentIndex', agentIndex)
        # self.lg('gameState.getLegalActions(agentIndex)', gameState.getLegalActions(agentIndex))
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            gs = gameState.generateSuccessor(agentIndex, action)
            depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)
            curr_val = self.val(gs, depth, newAgentIndex)
            v = min(v, curr_val)
        return v

    def get_new_agent(self, agentIndex, depth, gameState):
        newAgentIndex = agentIndex + 1
        if gameState.getNumAgents() == newAgentIndex:
            newAgentIndex = 0
        if newAgentIndex == 0:
            depth = depth + 1
        return depth, newAgentIndex

    def val(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            max_value = self.max_val(gameState, depth, agentIndex)
            return max_value
        else:
            min_value = self.min_val(gameState, depth, agentIndex)
            return min_value


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        nextActions = [(g, self.val(gameState, 0, 0)) for g in gameState.getLegalActions(0)]
        # self.lg('nextActions', nextActions)
        min_util = float('inf')
        if len(nextActions) > 0:
            min_util = min([a[1] for a in nextActions])
        # self.lg('min_util', min_util)
        for a in nextActions:
            if a[1] == min_util:
                return a[0]
        raise Exception("shud not reach here")

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
