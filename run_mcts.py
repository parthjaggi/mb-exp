# wolf traffic
from mcts.traffic import TrafficEnv
# from traffic import TrafficEnv

# loading saved model
from pathlib import Path
from tools import load_dataset, get_test_train_episodes, discrete_to_onehot
from main import load_config
from models.conv_model import ConvModel, ConvTransitionModel, ConvTransitionModel2, ConvTransitionModel3, ClassificationModel
from plotting import visualize_recons, visualize_recons_multi, cutoff_image, sl_to_ml, ml_to_sl
from tools import pformat
from bisect import bisect_left
import torch

# mcts
from mcts.monte_carlo_tree_search import MCTS, Node
from mcts.traffic2 import TrafficState, new_traffic_state, append_action_to_ph, append_phase_to_ph
# from monte_carlo_tree_search import MCTS, Node
# from traffic2 import TrafficState, new_traffic_state, append_action_to_ph, append_phase_to_ph


# get obs
env = TrafficEnv()
obs = env.reset()


# load model
model = ClassificationModel()
p = Path('./results/classification/2020-12-14--T22-48-12/best.tar')
state = torch.load(p)
model.load_state_dict(state['state_dict'])
model.cutoff_image = cutoff_image

# mcts
tree = MCTS(single_player=True)
sx = torch.Tensor(obs['dtse'][0, 3:, :, 0])
sp = torch.Tensor(obs['phase'][0, 3:, :, 0])
ts = new_traffic_state(sx, sp, model)


def get_traffic_state(obs):
    sx = torch.Tensor(obs['dtse'][0, 3:, :, 0])
    sp = torch.Tensor(obs['phase'][0, 3:, :, 0])
    ts = new_traffic_state(sx, sp, model)
    return ts

rollout_count = 20

while True:
    for _ in range(rollout_count):
        tree.do_rollout(ts)
    ts = tree.choose(ts)
    action = ts.action

    obs, reward, done, info = env.step(action)
    ts = get_traffic_state(obs)
    print('current ph', ts.ph.numpy(), reward)