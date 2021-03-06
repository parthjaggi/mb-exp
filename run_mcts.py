# wolf traffic
from mcts.traffic import TrafficEnv
# from traffic import TrafficEnv

# loading saved model
from pathlib import Path
from tools import load_dataset, get_test_train_episodes, discrete_to_onehot
from main import load_config
from models.conv_model import *
from plotting import visualize_recons, visualize_recons_multi, cutoff_image, sl_to_ml, ml_to_sl
from tools import pformat
from bisect import bisect_left
import torch
import argparse

# mcts
from mcts.monte_carlo_tree_search import MCTS, PureRollouts, Node
from mcts.traffic2 import TrafficState, new_traffic_state, append_action_to_ph, append_phase_to_ph, EXTEND, CHANGE
# from monte_carlo_tree_search import MCTS, Node
# from traffic2 import TrafficState, new_traffic_state, append_action_to_ph, append_phase_to_ph

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', action='store_true', default=False)
parser.add_argument('--static', type=str)
args = parser.parse_args()

# get obs
env = TrafficEnv()
obs = env.reset()


# load model
# model = ClassificationModel()
# p = Path('./results/classification/2020-12-14--T22-48-12/best.tar')
# p = Path('./results/classification/2021-01-26--T10-39-19/best.tar')

model = ClassificationModel2()
# p = Path('./results/classification/2021-01-26--T14-38-14/best.tar')
p = Path('./results/classification/2021-01-27--T08-05-26/best.tar')

# model = ClassificationModel3()
# p = Path('./results/2021-01-27--T09-23-39/best.tar')

# model = ConvTransitionModel2_2()
# p = Path('./results/2021-01-25--T19-51-36/best.tar')

state = torch.load(p)
model.load_state_dict(state['state_dict'])
model.cutoff_image = cutoff_image
model.cutoff = 0.5

# mcts or rollouts
tree = PureRollouts(single_player=True) if args.rollouts else MCTS(single_player=True)
sx = torch.Tensor(obs['dtse'][0, 3:, :, 0])
sp = torch.Tensor(obs['phase'][0, 3:, :, 0])
ts = new_traffic_state(sx, sp, model)


def get_traffic_state(obs):
    sx = torch.Tensor(obs['dtse'][0, 3:, :, 0])
    sp = torch.Tensor(obs['phase'][0, 3:, :, 0])
    ts = new_traffic_state(sx, sp, model)
    return ts

rollout_count = 20
episode_rewards = 0
done = False

def print_statistics():
    print('episode_rewards', episode_rewards)

if args.static:
    p1_time, p2_time = map(int, args.static.split(','))
    phase_times = {0: p1_time, 1: p2_time}
    while not done:
        phase = ts.ph[0, -1].item()
        if len(ts.legal_actions) > 1:
            action = CHANGE if ts.phase_time == phase_times[phase] else EXTEND
        else:
            ts = tree.choose(ts)
            action = ts.action
        obs, reward, done, info = env.step(action)
        ts = get_traffic_state(obs)
        episode_rewards += reward

    print_statistics()
    exit()


while not done:
    if len(ts.legal_actions) > 1:
        print('multiple legal actions')
        for _ in range(rollout_count):
            tree.do_rollout(ts)
    ts = tree.choose(ts)
    action = ts.action

    print('done', done)
    print('episode_rewards', episode_rewards)
    obs, reward, done, info = env.step(action)
    print('done', done)
    print('episode_rewards', episode_rewards)
    episode_rewards += reward
    ts = get_traffic_state(obs)
    # print('current ph', ts.ph.numpy(), reward)

print_statistics()
exit()