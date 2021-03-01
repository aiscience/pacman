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


import mdp, util

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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp

        for i in range(self.iterations):
            vk = self.values.copy()

            for state in mdp.getStates():
                self.update_state(mdp, state, vk)

            self.values = vk

    def update_state(self, mdp, state, vk):
        Q = util.Counter()
        for action in mdp.getPossibleActions(state):
            Q[action] = self.computeQValueFromValues(state, action)
        vk[state] = Q[Q.argMax()]

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

        mdp = self.mdp
        discount = self.discount
        sum = 0
        for (nextState, prob) in mdp.getTransitionStatesAndProbs(state, action):
          R = mdp.getReward(state, action, nextState)
          sum = sum + prob * (R + (discount * self.getValue(nextState)))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp

        selected_action = None
        Q_max = float('-inf')

        for action in mdp.getPossibleActions(state):
          Q = self.computeQValueFromValues(state, action)
          if Q > Q_max:
            Q_max = Q
            selected_action = action
        return selected_action

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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        curr_iter = 0
        mdp = self.mdp
        while curr_iter < self.iterations:
            for state in self.mdp.getStates():
                self.update_state(mdp, state, self.values)
                curr_iter += 1
                if curr_iter >= self.iterations:
                    return



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        predecessors = {}
        for state in states:
            predecessors[state] = set()

        # Initialize an empty priority queue.
        priority_queue = util.PriorityQueue()

        for state in states:
            Q = util.Counter()

            for action in self.mdp.getPossibleActions(state):
                # Compute predecessors of all states.
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob != 0:
                        predecessors[nextState].add(state)

                # For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
                # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
                # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                Q[action] = self.computeQValueFromValues(state, action)

            if not self.mdp.isTerminal(state):  # means: if non terminal state
                diff = abs(self.values[state] - Q[Q.argMax()])
                priority_queue.update(state, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priority_queue.isEmpty():
                return
            # Pop a state s off the priority queue.
            state = priority_queue.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
                Q = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    Q[action] = self.computeQValueFromValues(state, action)

                self.values[state] = Q[Q.argMax()]
            # For each predecessor p of s, do
            for p in predecessors[state]:
                predecessor_Q = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    predecessor_Q[action] = self.computeQValueFromValues(p, action)

                # Find the absolute value of the difference between the current value of p in self.values and the
                # highest Q-value across all possible actions from p (this represents what the value should be);
                # call this number diff. Do NOT update self.values[p] in this step.
                diff = abs(self.values[p] - predecessor_Q[predecessor_Q.argMax()])
                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative),
                # as long as it does not already exist in the priority queue with equal or lower priority.
                # As before, we use a negative because the priority queue is a min heap, but we want to prioritize
                # updating states that have a higher error.
                if diff > self.theta:
                    priority_queue.update(p, -diff)
