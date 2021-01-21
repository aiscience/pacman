# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import *


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    startState = problem.getStartState()
    s = Stack()
    s.push((startState, []))
    visited = set()
    while not s.isEmpty():
        current_path = s.pop()
        current_state = current_path[0]
        path_to_curr_state = current_path[1]
        if problem.isGoalState(current_state):
            return path_to_curr_state
        if current_state in visited:
            continue
        successors = problem.getSuccessors(current_state)
        for successor in successors:
            successor_state = successor[0]
            successor_direction = successor[1]
            new_directions = list(path_to_curr_state)
            new_directions.append(successor_direction)
            s.push((successor_state, new_directions))
        visited.add(current_state)
    raise Exception("Searched the maze but did not find the goal")



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    startState = problem.getStartState()
    s = Queue()
    s.push((startState, []))
    visited = set()
    i = 1
    while not s.isEmpty():
        current_path = s.pop()
        current_state = current_path[0]
        path_to_curr_state = current_path[1]
        if problem.isGoalState(current_state):
            return path_to_curr_state
        if current_state in visited:
            continue
        successors = problem.getSuccessors(current_state)
        for successor in successors:
            successor_state = successor[0]
            successor_direction = successor[1]
            new_directions = list(path_to_curr_state)
            new_directions.append(successor_direction)
            s.push((successor_state, new_directions))
        visited.add(current_state)
        i = i + 1
    raise Exception("Searched the maze but did not find the goal")


def uniformCostSearch(problem):
    startState = problem.getStartState()
    s = PriorityQueue()
    s.push((startState, []), 0)
    visited = set()
    while not s.isEmpty():
        current_path = s.pop()
        current_state = current_path[0]
        path_to_curr_state = current_path[1]
        if problem.isGoalState(current_state):
            return path_to_curr_state
        if current_state in visited:
            continue
        successors = problem.getSuccessors(current_state)
        for successor in successors:
            successor_state = successor[0]
            successor_direction = successor[1]
            new_directions = list(path_to_curr_state)
            new_directions.append(successor_direction)
            cost_to_successor = problem.getCostOfActions(new_directions)
            s.update((successor_state, new_directions), cost_to_successor)
        visited.add(current_state)
    raise Exception("Searched the maze but did not find the goal")

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    startState = problem.getStartState()
    s = PriorityQueue()
    start_state_h = heuristic(startState, problem)
    s.push((startState, []), start_state_h)
    visited = set()
    while not s.isEmpty():
        current_path = s.pop()
        current_state = current_path[0]
        path_to_curr_state = current_path[1]
        if problem.isGoalState(current_state):
            return path_to_curr_state
        if current_state in visited:
            continue
        successors = problem.getSuccessors(current_state)
        for successor in successors:
            successor_state = successor[0]
            successor_direction = successor[1]
            new_directions = list(path_to_curr_state)
            new_directions.append(successor_direction)
            cost_to_successor = problem.getCostOfActions(new_directions) + heuristic(successor_state, problem)
            s.update((successor_state, new_directions), cost_to_successor)
        visited.add(current_state)
    raise Exception("Searched the maze but did not find the goal")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
