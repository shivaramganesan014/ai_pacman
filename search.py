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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    pathTaken = []
    pathTaken.append((problem.getStartState(), []))

    return dfsFunc(pathTaken, [], problem)

def dfs_search_with_stack(problem):
    visited = []
    stack = util.Stack()
    stack.push((problem.getStartState(), []))

    while(not stack.isEmpty()):
        state, directions = stack.pop()
        if (not state in visited):
            visited.append(state)
            if problem.isGoalState(state):
                return directions
            # directions.append(state[1])
            for n_state in problem.getSuccessors(state):
                nextInstruction = directions + [n_state[1]]
                stack.push((n_state[0], nextInstruction))
    return []

def  dfs_search_recursion(problem, visited, current_state, directions):
    from game import Directions
    for state in problem.getSuccessors(current_state):
        new_pos = state[0]
        direction = state[1]
        if problem.isGoalState(current_state):
            return directions
        if new_pos in visited:
            continue
        # visited.append(new_pos)
        visited.append(new_pos)
        directions.append(direction)
        dfs_search_recursion(problem, visited, new_pos, directions);
        directions.append(Directions.REVERSE[direction])
        visited.remove(new_pos)
    return directions

def dfsFunc(pathTaken, visited, problem):
    currentNode = pathTaken.pop()
    if problem.isGoalState(currentNode[0]):
        # print(pathTaken)
        # print(len(currentNode[1]))
        return currentNode[1]
    elif visited.count(currentNode[0]) == 0:
        visited.append(currentNode[0])
        for moves in problem.getSuccessors(currentNode[0]):
            if visited.count(moves[0]) == 0:
                pathTaken.append((moves[0], currentNode[1] + [moves[1]]))
        return dfsFunc(pathTaken, visited, problem)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    queue = util.Queue()
    # queue.push((problem.getStartState(), []))
    return bfs_search(problem, queue, [])

def bfs_search(problem, queue, visited):
    possible_successors = []
    queue.push((problem.getStartState(), possible_successors))
    while(not queue.isEmpty()):
        current_pos, path = queue.pop()
        if current_pos not in visited:
            visited.append(current_pos)
            if problem.isGoalState(current_pos):
                return path
            for successor in problem.getSuccessors(current_pos):
                possible_successors = path + [successor[1]]
                queue.push((successor[0], possible_successors))
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    queue = util.PriorityQueue()
    # print problem.getCostOfActions(["West", "South"])
    return ucs_search(problem, queue, [], None, False)

def astar_search(problem, queue, visited, heuristic):
    return ucs_search(problem, queue, visited ,heuristic, applyHeuristic=True)

def ucs_search(problem, queue, visited, heuristic ,applyHeuristic = False):
    # import searchAgents as sa
    possible_successors = []
    queue.push((problem.getStartState(), possible_successors), 0)
    while (not queue.isEmpty()):
        current_pos, path = queue.pop()
        if current_pos not in visited:
            visited.append(current_pos)
            if problem.isGoalState(current_pos):
                return path
            for successor in problem.getSuccessors(current_pos):
                possible_successors = path + [successor[1]]
                cost = problem.getCostOfActions(possible_successors)
                if(applyHeuristic):
                    # print sa.manhattanHeuristic(current_pos, problem)
                    cost += heuristic(successor[0], problem)
                # print cost
                queue.push((successor[0], possible_successors), cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return astar_search(problem, util.PriorityQueue(), [], heuristic)
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
