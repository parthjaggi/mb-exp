import torch
import numpy as np
from collections import namedtuple
from random import choice
from bisect import bisect_left
from mcts.monte_carlo_tree_search import MCTS, Node

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")
_TS = namedtuple("TrafficState", "ls ph")  # lane_state and phase_history

EXTEND = 0
CHANGE = 1



# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TrafficState(_TS, Node):
    def __init__(self, ls, ph):
        super().__init__()
        self.init_phase_info()
        self.model = None
        self.action = None
        self.reward_val = None
        self.terminal = False
        self.legal_actions = self.get_legal_actions()

    def find_children(self):
        if self.terminal:
            return set()
        return {self.take_action(act) for act in self.legal_actions}

    def get_legal_actions(self):
        if self.phase_time < 10:
            return [EXTEND]
        if self.phase_time == 60:
            return [CHANGE]
        return [EXTEND, CHANGE]

    def init_phase_info(self):
        self.current_phase = self.ph[:, -1:]
        timesteps = 0
        for phase in reversed(self.ph[0]):
            if phase != self.current_phase[0, 0]:
                break
            timesteps += 1
        self.phase_time = timesteps

    def find_child_for_simulation(self, change_at_phase_time):
        if self.terminal:
            return None
        if len(self.legal_actions) > 1:
            action = CHANGE if self.phase_time > change_at_phase_time else EXTEND
        else:
            action = next(iter(self.legal_actions))
        return self.take_action(action)

    def find_random_child(self):
        if self.terminal:
            return None
        # choose randomly from legal actions
        return self.take_action(choice(self.legal_actions))

    def reward(self):
        if not self.reward_val:
            self.reward_val = self._reward()
        return self.reward_val
    
    def _reward(self):
        # Get next state by EXTENDing current state.
        # Get speeds, and estimate reward.
        ph = append_action_to_ph(self.ph, EXTEND)
        next_state = self.get_next_state(self.ls, ph)
        veh_info = get_veh_info(self.ls, next_state)
        print(veh_info)
        
        reward = queue_reward(veh_info)
        if reward != 0:
            print(self.action, reward)
        return reward

    def is_terminal(self):
        return self.terminal
    
    def get_next_state(self, ls, ph):
        ls2 = self.model(ls, ph)
        ls2 = self.model.cutoff_image(ls2, 0.5, minmax=[0, 1])
        return ls2

    def take_action(self, action):
        # check if action is legal.
        assert action in self.legal_actions
        
        ph = append_action_to_ph(self.ph, action)
        next_state = self.get_next_state(self.ls, ph)
        return new_traffic_state(next_state, ph, self.model, action)

    # def take_action(self, index):
    #     tup = self.tup[:index] + (self.turn,) + self.tup[index + 1 :]
    #     turn = not self.turn
    #     winner = _find_winner(tup)
    #     is_terminal = (winner is not None) or not any(v is None for v in tup)
    #     return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def new_traffic_state(sx, sp, model, action=None):
    ts = TrafficState(sx, sp)
    ts.model = model
    ts.action = action
    return ts

def append_action_to_ph(ph, action):
    current_phase = ph[:, -1]
    phase = current_phase if action == EXTEND else 1 - current_phase
    return append_phase_to_ph(ph, phase)

def append_phase_to_ph(ph, phase):
    return torch.cat((ph, phase.reshape(1, 1)), 1)[:, 1:]

def get_veh_info(ls, ls2):
    t1_idxs = torch.where(ls[0] > 0)[0]
    t2_idxs = torch.where(ls2[0] > 0)[0]
    veh_info = {}

    for veh_id, veh_pos in enumerate(t1_idxs):
        veh_info[veh_id] = {'t1': veh_pos.item(), 't2': None}

    for veh_pos in t2_idxs:
        veh_id = closest_array_id(t1_idxs, veh_pos)
        if veh_id < len(t1_idxs):
            veh_info[veh_id]['t2'] = veh_pos.item()
        else:
            veh_info[veh_id] = {'t1': None, 't2': veh_pos.item()}
    return veh_info

def queue_reward(veh_info):
    reward = 0
    for veh_id, veh_dict in veh_info.items():
        if veh_dict['t1'] != None and veh_dict['t2'] != None:
            speed = veh_dict['t1'] - veh_dict['t2']  # m/s
            veh_dict['speed'] = speed
            reward -= (speed < 1)
    return reward

def closest_array_id(array, val):
    return bisect_left(array.cpu().numpy(), val)