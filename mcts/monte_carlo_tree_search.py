"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np

EXTEND = 0
CHANGE = 1


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, single_player=False, gamma=1, sim_horizon=10):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node

        self.gamma = gamma
        self.sim_horizon = sim_horizon
        self.single_player = single_player
        self.exploration_weight = exploration_weight
        self._simulate = self._simulate_sp2 if single_player else self._simulate_mp
        self._backpropagate = self._backpropagate_sp if single_player else self._backpropagate_mp

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        print('choose', node.current_phase, list(map(score, self.children[node])), list(map(lambda x: x.action, self.children[node])), max(self.children[node], key=score).action)

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward, action_sequence = self._simulate(leaf)
        if reward < 0:
            print('reward', reward, 'action_sequence', action_sequence)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    # sp: single player
    # mp: multi player
    def _simulate_sp(self, node):
        # sp: returns the discounted_reward, action_sequence.
        "Returns the reward for a random simulation (to completion) of `node`"
        steps, discounted_reward = 0, 0
        action_sequence = []
        while True:
            if steps >= self.sim_horizon or node.is_terminal():
                return discounted_reward, action_sequence
            node = node.find_random_child()
            action_sequence.append(node.action)
            steps += 1
            discounted_reward += (self.gamma ** (steps)) * node.reward()

    def _simulate_sp2(self, node):
        # sp2: returns the discounted_reward, action_sequence.
        # created from sp, so that rollouts are uniform sampling from allowed timesteps.
        # currently being used.
        "Returns the reward for a random simulation (to completion) of `node`"
        steps, discounted_reward = 0, 0
        action_sequence = []
        change_at_phase_time = np.random.randint(10, 60)

        while True:
            if steps >= self.sim_horizon or node.is_terminal():
                return discounted_reward, action_sequence
            node = node.find_child_for_simulation(change_at_phase_time)
            if node.action == CHANGE:
                change_at_phase_time = np.random.randint(10, 60)
            action_sequence.append(node.action)
            steps += 1
            discounted_reward += (self.gamma ** (steps)) * node.reward()

    def _simulate_sp3(self, node):
        # sp3: returns the sequence of rewards.
        # oldest. likely to be deleted.
        "Returns the reward for a random simulation (to completion) of `node`"
        steps, rewards = 0, []
        while True:
            if steps >= self.sim_horizon or node.is_terminal():
                return rewards
            node = node.find_random_child()
            rewards.append(node.reward())
            steps += 1

    def _simulate_mp(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True  # if game ends now, human action caused it and it needs to be inverted for machine.
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate_sp(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        # TODO: recheck this method.
        discounted_rewards = reward
        print('_backpropagate reward', reward)
        previous_node = None  # previous with respect to the reversed path loop below. it is next with respect to the forward path.
        for idx, node in enumerate(reversed(path)):
            reward = previous_node.reward() if previous_node else 0
            discounted_rewards += reward + discounted_rewards * (self.gamma)
            self.N[node] += 1
            self.Q[node] += discounted_rewards
            previous_node = node

    def _backpropagate_sp2(self, path, rewards):
        "Send the reward back up to the ancestors of the leaf"
        discounted_rewards = 0
        for idx, node in enumerate(reversed(path)):
            discounted_rewards += rewards[-1 - idx] + discounted_rewards * (self.gamma ** idx)
            self.N[node] += 1
            self.Q[node] += discounted_rewards

    def _backpropagate_mp(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)



class PureRollouts:
    "At each state, perform rollouts for each action to choose an action."

    def __init__(self, single_player=False, gamma=1, sim_horizon=30, debug=True):
        self.reset()
        self.debug = debug
        self.gamma = gamma
        self.sim_horizon = sim_horizon
        self.single_player = single_player
        self._simulate = self._simulate_single_player2 if single_player else self._simulate_multi_player
        self._backpropagate = self._backpropagate_single_player if single_player else self._backpropagate_multi_player

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        max_value_node = max(self.children[node], key=score)

        if self.debug:
            print('choose debug.')
            print('current phase:', node.current_phase)
            print('child scores:', list(map(score, self.children[node])))
            print('child actions:', list(map(lambda x: x.action, self.children[node])))
            print('best action:', max_value_node.action)

        self.reset()
        return max_value_node
    
    def reset(self):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node

    def do_rollout(self, node):
        """
        If n legal actions, do 1 rollout for each legal action.
        Do rollout from the child node reached from current node by doing legal action.
        """
        path = [node]
        leaf = path[-1]

        if self.debug:
            print('********************** do rollout **********************')
            self.decribe_state(leaf)

        self._expand(leaf)
        for c in self.children[leaf]:
            self.decribe_state(c)
            reward, action_sequence = self._simulate(c)
            if reward < 0 and self.debug:
                print('reward', reward, 'action_sequence', action_sequence)
            self._backpropagate(path + [c], reward)


    def decribe_state(self, node):
        print()
        print('state', node.ls)
        print('phase', node.ph)
        print('action', node.action)
        print('phase_time', node.phase_time)
        pass

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate_single_player2(self, node):
        # sp2: returns the discounted_reward, action_sequence.
        # created from sp, so that rollouts are uniform sampling from allowed timesteps.
        "Returns the reward for a random simulation (to completion) of `node`"
        steps, discounted_reward = 0, 0
        action_sequence = []
        change_at_phase_time = np.random.randint(10, 60)
        if self.debug: print('change_at_phase_time', change_at_phase_time)

        while True:
            if steps >= self.sim_horizon or node.is_terminal():
                return discounted_reward, action_sequence
            node = node.find_child_for_simulation(change_at_phase_time)
            if node.action == CHANGE:
                change_at_phase_time = np.random.randint(10, 60)
                if self.debug: print('change_at_phase_time', change_at_phase_time)
            action_sequence.append(node.action)
            steps += 1
            discounted_reward += (self.gamma ** (steps)) * node.reward()

    def _backpropagate_single_player(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        # TODO: recheck this method.
        discounted_rewards = reward
        print('_backpropagate reward', reward)
        previous_node = None  # previous with respect to the reversed path loop below. it is next with respect to the forward path.
        for idx, node in enumerate(reversed(path)):
            reward = previous_node.reward() if previous_node else 0
            discounted_rewards += reward + discounted_rewards * (self.gamma)
            self.N[node] += 1
            self.Q[node] += discounted_rewards
            previous_node = node


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True