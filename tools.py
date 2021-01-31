import pathlib
import numpy as np
import functools
import tensorflow as tf
import torch
from tensorflow.keras.mixed_precision import experimental as prec
import datetime
from decimal import Decimal


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5  # change 255 to 1. as our obs have max of 1.
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs

def preprocess2(obs, config):
    # dtype = prec.global_policy().compute_dtype
    # obs = obs.copy()
    # with tf.device('cpu:0'):
    #     obs['image'] = tf.cast(obs['image'], dtype) - 0.5
    #     obs['phase'] = tf.cast(obs['phase'], dtype) - 0.5
    #     clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    #     obs['reward'] = clip_rewards(obs['reward'])
    return obs


def load_dataset(config, eps):
    directory = config.datadir
    episode = next(load_episodes(directory, 1, eps=eps))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    batch_length = config.batch_length + 1 if config.type == 'dream' else config.batch_length
    generator = lambda: load_episodes(directory, config.train_steps, batch_length, config.dataset_balance, eps=eps)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess2, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def load_episodes(directory, rescan, length=None, balance=False, seed=0, eps=[]):
    """
    Args:
        directory (pathlib.PosixPath): directory.
        rescan (int): rescan is number of episode_ids needed from episode list.
        length (int, optional): Length of sample. Defaults to None.
        balance (bool, optional): Defaults to False.
        seed (int, optional): Defaults to 0.

    Yields:
        dict: episode sample dictionary.
    """
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}  # cache has all episode files
    while True:
        # for filename in directory.glob('*.npz'):
        for filename in eps:
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        # episode = np.load(f)
                        episode = np.load(f, allow_pickle=True)
                        episode = episode.item()
                        if 'obs' in episode:
                            episode['x'] = episode['obs']
                            del episode['obs']
                        if 'phase' in episode:
                            episode['phases'] = episode['phase']
                            del episode['phase']
                        
                        ## below: commented danijar code
                        # episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 1:
                    print(f'Skipped short episode of length {available}.')
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(random.randint(0, available))
                episode = {k: v[index : index + length] for k, v in episode.items()}
            yield episode

def get_sample(id):
    # given id, find episode, and then find batch
    # 
    pass

def num_of_sequence_samples(config):
    # returns the number of samples that can be created.
    # based on number of episodes, their size and the batch length.
    episode_size = 500
    num_episodes = len(config.datadir.glob('*.npz'))
    return (episode_size - config.batch_length) * num_episodes

def get_test_train_episodes(config):
    # divide episodes into train and test. last 2 episodes are test.
    filenames = list(config.datadir.glob('*.npy'))
    # train_eps = filenames[:-2]
    # test_eps = filenames[-2:]
    test_eps = [i for i in filenames if '8.npy' in str(i) or '9.npy' in str(i)]
    train_eps = [i for i in filenames if i not in test_eps]
    return train_eps, test_eps

def discrete_to_onehot(y, limit):
    y_onehot = torch.FloatTensor(y.shape[0], limit)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def get_timestamp(ts_format='%Y-%m-%d--T%H-%M-%S'):
    return datetime.datetime.now().strftime(ts_format)

class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


def pformat(x, digits=8):
    # makes loss tensors like 4e-6 more readable when printing.
    return round(Decimal(x.item()), digits)
