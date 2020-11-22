import pathlib
import numpy as np
import functools
import tensorflow as tf


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


def load_dataset(directory, config):
    episode = next(load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: load_episodes(directory, config.train_steps, config.batch_length, config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
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
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
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

