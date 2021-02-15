import torch
import numpy as np
from collections import defaultdict, namedtuple
from random import choice
from bisect import bisect_left
from mcts.monte_carlo_tree_search import MCTS, Node
from flow.core.util import get_phase_action
from utils import first

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")
_TS = namedtuple("TrafficState", "ls ph ss")  # lane_state, phase_history, speed_state

EXTEND = 0
CHANGE = 1



# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TrafficState(_TS, Node):
    def __new__(cls, ls, ph, ss=None):
        """
        Allows ss to not be omitted during initialization, and assigns it None.
        """
        self = super().__new__(cls, ls, ph, ss)
        return self

    def __init__(self, ls, ph, ss=None):
        super().__init__()
        self.debug = True
        self.init_phase_info2()
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

    def init_phase_info2(self):
        # current phase is based on WE lane, NS lane phase ignored for now.
        self.current_phase = self.ph[1:2, -1:]  # shape change from (1, 60) to (2, 60).
        timesteps = 0
        for phase in reversed(self.ph[1]):  # self.ph[0]: for sl.
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
            action = first(self.legal_actions)
        return self.take_action(action)

    def find_child_for_simulation2(self):
        # PURELY EXTEND.
        if self.terminal:
            return None
        if len(self.legal_actions) > 1:
            action = EXTEND
        else:
            action = first(self.legal_actions)
        return self.take_action(action)

    def find_child_for_simulation3(self, is_change):
        if self.terminal:
            return None
        if len(self.legal_actions) > 1:
            action = CHANGE if is_change else EXTEND
        else:
            action = first(self.legal_actions)
        return self.take_action(action)

    def find_random_child(self):
        if self.terminal:
            return None
        # choose randomly from legal actions
        return self.take_action(choice(self.legal_actions))

    def reward(self):
        if not self.reward_val:
            self.reward_val = self._reward2()
        return self.reward_val
    
    def _reward(self):
        # Get next state by EXTENDing current state.
        # Get speeds, and estimate reward.
        ph = append_action_to_ph(self.ph, EXTEND)
        next_state = self.get_next_state(self.ls, self.ph, EXTEND)
        veh_info = get_veh_info(self.ls, next_state)
        if self.debug: print(veh_info)
        
        reward = queue_reward(veh_info)
        if reward != 0 and self.debug:
            print(ph[:, -1], self.action, reward)
        return reward
    
    def _reward2(self):
        reward, lane_rewards = self._reward_from_state_predictions() if self.ss is None else self._reward_from_speeds()

        if reward != 0 and self.debug:
            print('negative reward info --', 'lane_rewards:', lane_rewards)
        return reward
    
    def _reward_from_state_predictions(self):
        # Get next state by EXTENDing current state.
        # Get speeds, and estimate reward.
        ph = append_action_to_ph2(self.ph, EXTEND)
        next_state, _ = self.get_next_state2(self.ls, self.ss, self.ph, ph, EXTEND)

        lane_rewards = {}
        for idx in range(self.ls.shape[0]):
            veh_info = get_veh_info(self.ls[idx:idx+1], next_state[idx:idx+1])
            # if self.debug: print(veh_info)
            lane_rewards[idx] = queue_reward(veh_info)

        reward = sum(lane_rewards.values())
        return reward, lane_rewards
    
    def _reward_from_speeds(self):
        NORM_SPEED, STOP_SPEED = 35, 2  # STOP_SPEED=2 in sow45_code3.
        lane_rewards = defaultdict(int)
        veh_positions = torch.where(self.ls > 0)

        for r, c in (zip(*veh_positions)):
            veh_speed = self.ss[r, c]
            reward = -1 if veh_speed * NORM_SPEED < STOP_SPEED else 0
            lane_rewards[r.item()] += reward
        return sum(lane_rewards.values()), lane_rewards

    def is_terminal(self):
        return self.terminal
    
    def get_next_state(self, ls, ph, action):
        action = torch.Tensor([action]).reshape(1, -1)
        phase_action = torch.Tensor(get_phase_action(ph[0, -1].item(), action)).reshape(1, -1)
        sample = {'x': ls, 'phases': ph, 'action': action, 'phase_action': phase_action}
        ls2 = self.model(sample)  # TODO: is the preprocessing here, same as what recieved during model training?
        ls2 = self.model.cutoff_image(ls2, self.model.cutoff, minmax=[0, 1])
        return ls2
    
    def get_next_state2(self, ls, ss, ph, next_ph, action):
        """
        get_next_state2 builds on get_next_state.
        Created for multi-lane next state prediction and vehicle-model usage.
        """
        assert ls.shape[0] == ph.shape[0]
        action = torch.Tensor([action]).reshape(1, -1)
        next_ls_list = []
        next_ss_list = []
        num_lanes = ls.shape[0]

        for idx in range(num_lanes):
            phase_action = torch.Tensor(get_phase_action(ph[idx:idx+1, -1].item(), action)).reshape(1, -1)
            lane_ss = ss[idx:idx+1] if ss is not None else None
            sample = {'x': ls[idx:idx+1], 'v': lane_ss, 'phases': ph[idx:idx+1], 'next_phases': next_ph[idx:idx+1], 'action': action, 'phase_action': phase_action}
            next_ls = self.model(sample)
            next_ls, next_ss = (next_ls[0], next_ls[1]) if type(next_ls) is tuple else (next_ls, None)  # for vehicle models, (ls, ss) is returned.
            next_ls = self.model.cutoff_image(next_ls, self.model.cutoff, minmax=[0, 1])
            next_ls_list.append(next_ls)
            next_ss_list.append(next_ss)

        next_ls = torch.cat(next_ls_list).reshape(num_lanes, -1)
        next_ss = torch.cat(next_ss_list).reshape(num_lanes, -1) if next_ss is not None else None
        return next_ls, next_ss

    def take_action(self, action):
        # check if action is legal.
        assert action in self.legal_actions
        
        ph = append_action_to_ph2(self.ph, action)
        next_state, next_ss = self.get_next_state2(self.ls, self.ss, self.ph, ph, action)
        return new_traffic_state(next_state, ph, next_ss, self.model, action)

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

def new_traffic_state(sx, sp, ss, model, action=None):
    ts = TrafficState(sx, sp, ss)
    ts.model = model
    ts.action = action
    return ts

def append_action_to_ph(ph, action):
    current_phase = ph[:, -1]
    phase = current_phase if action == EXTEND else 1 - current_phase
    return append_phase_to_ph(ph, phase)

def append_action_to_ph2(ph, action):
    # get next phase for 2 lane case.
    ph_NS = append_action_to_ph(ph[0:1], action)
    ph_WE = append_action_to_ph(ph[1:2], action)
    return torch.cat((ph_NS, ph_WE))

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