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
    a = problem.getSuccessors(problem.getStartState()) 
    print "Start's successors:", a
    print "only one", len(a)
    """
    visited = set()
    node_list = util.Stack()
    start_node = problem.getStartState()
    moves = []
    init_cost = 0
    node_list.push((start_node, moves, init_cost))
    
    while node_list.isEmpty() is False:
        node_pos, node_move, node_cost = node_list.pop()
        if node_pos not in visited:
    
            visited.add(node_pos)
        
            if problem.isGoalState(node_pos):
                return node_move
        
            successors = problem.getSuccessors(node_pos)
            for node in successors:
                node_list.push((node[0], node_move + [node[1]] , node[2]))
    
    #print "mov", node_move
    
    return []


    # 1er lloc on la cridem:
    #	sempre li passem el problem
    # 	El primer cop li passem el node inicial amb un array de direccions (buit) i el cost acumulat (zero)
    #dfs_loop(problem, current_node, [])
    
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    visited = set()
    node_list = util.Queue()
    start_node = problem.getStartState()
    moves = []
    init_cost = 0
    node_list.push((start_node, moves, init_cost))
    
    while node_list.isEmpty() is False:
        node_pos, node_move, node_cost = node_list.pop()
        if node_pos not in visited:
        
            visited.add(node_pos)
            
            if problem.isGoalState(node_pos):
                return node_move
            
            successors = problem.getSuccessors(node_pos)
            
            for node in successors:
                node_list.push((node[0], node_move + [node[1]] , node[2]))
    
    #print "mov", node_move
    
    return []
            
            
            
            
    
    #util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    node_list = util.PriorityQueue() # has 2 arguments, item and Priority
    visited = set()
    # same starting values as before
    start_node = problem.getStartState()
    moves = []
    init_cost = 0
    node_list.push((start_node, moves, init_cost), nullHeuristic(start_node, problem))
    
    while node_list.isEmpty() is False:
    
        node_pos, node_move, node_cost= node_list.pop()#poping does not return priority (second argument in push)

        if node_pos not in visited:
        
            visited.add(node_pos)
            
            if problem.isGoalState(node_pos):
                return node_move
            
            successors = problem.getSuccessors(node_pos)
            
            for node in successors:
                # the priority is calculated by getting the cost of the actions in the current node and adding the cost of the heuristic. 
                # We update the priority queue with that value. The smaller the value, the highest the priority, so we are always getting the calculated way with the least cost
                cost = problem.getCostOfActions(node_move) + nullHeuristic(node_pos, problem)
                node_list.update((node[0], node_move + [node[1]] , node[2]), cost)
            
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    node_list = util.PriorityQueue() # has 2 arguments, item and Priority
    visited = set()
    # same starting values as before
    start_node = problem.getStartState()
    moves = []
    init_cost = 0
    node_list.push((start_node, moves, init_cost), heuristic(start_node, problem))
    
    while node_list.isEmpty() is False:
    
        node_pos, node_move, node_cost= node_list.pop()#poping does not return priority (second argument in push)

        if node_pos not in visited:
        
            visited.add(node_pos)
            
            if problem.isGoalState(node_pos):
                return node_move
            
            successors = problem.getSuccessors(node_pos)
            
            for node in successors:
                # the priority is calculated by getting the cost of the actions in the current node and adding the cost of the heuristic. 
                # We update the priority queue with that value. The smaller the value, the highest the priority, so we are always getting the calculated way with the least cost
                cost = problem.getCostOfActions(node_move) + heuristic(node_pos, problem)
                node_list.update((node[0], node_move + [node[1]] , node[2]), cost)
            
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
