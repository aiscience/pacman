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

    def get_manhattan_distance(self, xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def get_euclidean_distance(self, xy1, xy2):
        return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

    def mean(self, ls):
        return sum(ls)/len(ls) if len(ls) > 0 else 0

    def scoreOnPosFoodGhost(self, position, foodGrid, ghostPos):
        total_food_cost = 0
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
        if total_food_cost == 0:
            return 99999999999
        # return 10
        return (1/total_food_cost)*100 if min_ghost_dist > 1 else -1

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        score = self.scoreOnPosFoodGhost(newPos, newFood, successorGameState.getGhostPositions())
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

def lg(k,v):
    return
    log=True
    if log:
        print("{k}={v}".format(k=k, v=str(v)))

class MinimaxAgent(MultiAgentSearchAgent):
    def max_val(self, gameState, depth, agentIndex):
        v = float('-inf')
        new_depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)
        for action in gameState.getLegalActions(agentIndex):
            gs = gameState.generateSuccessor(agentIndex, action)
            curr_val = self.val(newAgentIndex, new_depth, gs)
            v = max(v, curr_val)
            lg('maxv', v)
        return v

    def min_val(self, gameState, depth, agentIndex):
        v = float('inf')
        new_depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)
        for action in gameState.getLegalActions(agentIndex):
            gs = gameState.generateSuccessor(agentIndex, action)
            curr_val = self.val(newAgentIndex, new_depth, gs)
            v = min(v, curr_val)
            lg('minv', v)
        return v

    def get_new_agent(self, agentIndex, depth, gameState):
        newAgentIndex = agentIndex + 1
        if gameState.getNumAgents() == newAgentIndex:
            newAgentIndex = 0
        if newAgentIndex == 0:
            depth = depth + 1
        lg('depth and agentindex', [depth, newAgentIndex])
        return depth, newAgentIndex

    def val(self, agentIndex, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth+1:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            max_value = self.max_val(gameState, depth, agentIndex)
            return max_value
        else:
            min_value = self.min_val(gameState, depth, agentIndex)
            return min_value

    def getAction(self, gameState):
        int_min = float("-inf")
        action = None
        lg('gameState.getLegalActions(0)', gameState.getLegalActions(0))
        new_depth, newAgentIndex = self.get_new_agent(0, 0, gameState)
        for curr_action in gameState.getLegalActions(0):
            utility = self.val(newAgentIndex, new_depth, gameState.generateSuccessor(0, curr_action))
            lg('utility', utility)
            if utility > int_min:
                int_min = utility
                action = curr_action

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    def max_val(self, gameState, depth, agentIndex, alpha, beta):
        v = float("-inf")
        for newState in gameState.getLegalActions(agentIndex):
            v = max(v, self.val(1, depth, gameState.generateSuccessor(agentIndex, newState), alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def get_new_agent(self, agentIndex, depth, gameState):
        newAgentIndex = agentIndex + 1
        if gameState.getNumAgents() == newAgentIndex:
            newAgentIndex = 0
        if newAgentIndex == 0:
            depth = depth + 1
        lg('depth and agentindex', [depth, newAgentIndex])
        return depth, newAgentIndex

    def min_val(self, agentIndex, depth, gameState, alpha, beta):
        v = float("inf")
        new_depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)
        for newState in gameState.getLegalActions(agentIndex):
            v = min(v, self.val(newAgentIndex, new_depth, gameState.generateSuccessor(agentIndex, newState), alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def val(self, agentIndex, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:  # maximize for pacman
            return self.max_val(gameState, depth, agentIndex, alpha, beta)
        else:  # minimize for ghosts
            return self.min_val(agentIndex, depth, gameState, alpha, beta)

    def getAction(self, gameState):
        utility = float("-inf")
        action = None
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            ghostUtil = self.val(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if ghostUtil > utility:
                utility = ghostUtil
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):

    def get_new_agent(self, agentIndex, depth, gameState):
        newAgentIndex = agentIndex + 1
        if gameState.getNumAgents() == newAgentIndex:
            newAgentIndex = 0
        if newAgentIndex == 0:
            depth = depth + 1
        lg('depth and agentindex', [depth, newAgentIndex])
        return depth, newAgentIndex

    def max_val(self, gameState, depth, agentIndex):
        v = float("-inf")
        for newState in gameState.getLegalActions(agentIndex):
            v = max(v, self.val(1, depth, gameState.generateSuccessor(agentIndex, newState)))
        return v

    def expect_val(self, agentIndex, depth, gameState):
        v = float("inf")
        new_depth, newAgentIndex = self.get_new_agent(agentIndex, depth, gameState)

        v = sum(self.val(newAgentIndex, new_depth, gameState.generateSuccessor(agentIndex, newState)) for newState in
            gameState.getLegalActions(agentIndex)) / float(len(gameState.getLegalActions(agentIndex)))
        return v

    def val(self, agentIndex, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.max_val(gameState, depth, agentIndex)
        else:
            return self.expect_val(agentIndex, depth, gameState)

    def getAction(self, gameState):
        v = float("-inf")
        action = None
        for agentState in gameState.getLegalActions(0):
            utility = self.val(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > v or v == float("-inf"):
                v = utility
                action = agentState
        return action


def get_manhattan_distance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def get_euclidean_distance(xy1, xy2):
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

def scoreOnPosFoodGhost(position, foodGrid, ghostPos, gameState):
    import random
    # total_wall_cost = 1
    total_food_cost = 0
    # for w in wallList:
    #     total_wall_cost = total_wall_cost + self.get_euclidean_distance(w, position)
    foodList = foodGrid.asList()
    food_distances = [get_manhattan_distance(f, position) for f in foodList]
    food_distances.sort(reverse=True)
    factor = 1

    for f in food_distances:
        total_food_cost = total_food_cost + (f*factor)
        factor = factor * 10

    total_ghost_pos = 0.0
    min_ghost_dist = float('inf')
    for gp in ghostPos:
        ghost_dist = get_manhattan_distance(gp, position)
        min_ghost_dist = min(min_ghost_dist, ghost_dist)
        total_ghost_pos = total_ghost_pos + ghost_dist
    # total_ghost_pos = (total_ghost_pos * len(foodList) )**3
    lg("total_food_cost",total_food_cost)
    lg("total_ghost_pos",total_ghost_pos)
    if total_food_cost == 0:
        return float('inf')
    capsules = gameState.getCapsules()
    min_capsule_dist = float('inf')
    for gp in capsules:
        cap_dist = get_euclidean_distance(gp, position)
        min_capsule_dist = min(min_capsule_dist, cap_dist)

    if min_ghost_dist <= 1:
        return float('-inf')

    if min_capsule_dist <= 1:
        if random.randint(1, 4) == 3:
            return float('inf')

    return (1 / float(total_food_cost))

def betterEvaluationFunction1(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    ghost_pos = currentGameState.getGhostPositions()

    return scoreOnPosFoodGhost(newPos, newFood, ghost_pos, currentGameState)

def betterEvaluationFunction(currentGameState):
    import random
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    min_food_distance = -1

    food_distances = [get_manhattan_distance(f, newPos) for f in newFoodList]
    food_distances.sort(reverse=True)
    food_distances = food_distances[:4]
    factor = 1
    total_food_cost=0
    for f in food_distances:
        total_food_cost = total_food_cost + (f*factor)
        factor = factor * 2


    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if min_food_distance >= distance or min_food_distance == -1:
            min_food_distance = distance

    distances_to_ghosts = 1
    proximity_to_ghosts = 0
    min_ghost_dist = float('inf')
    for ghost_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost_state)
        min_ghost_dist = min(min_ghost_dist, distance)
        distances_to_ghosts += distance
        if distance <= 1:
            proximity_to_ghosts += 1

    """Obtaining the number of capsules available"""
    newCapsule = currentGameState.getCapsules()
    numberOfCapsules = len(newCapsule)

    if min_ghost_dist <= 1:
        return float('-inf')
    """Combination of the above calculated metrics."""
    return currentGameState.getScore()*2 + (1 / float(total_food_cost if total_food_cost > 0 else 1)) - proximity_to_ghosts - numberOfCapsules + 1/random.randint(1,3)

def betterEvaluationFunction1(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
