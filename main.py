import logging
import ray
import pprint
import yaml
import pathlib

from tools import AttrDict
from argparser import parser
from models.base_model import Model



def run():
    config = load_config('configs/dreamer.yaml')
    datadir = config.logdir / 'episodes'

    model = Model(config, datadir)

    train(model, config)


    # choose and load model

    # training loop

        # shuffled batches make more sense than sampling randomly from the dataset.
            # then we can have notion of epochs

        # sample from the episodes
            # look how dreamer samples from the disk

        # eval after every log_interval

        # video log after every video_log_interval

    # save model progress repeatedly

    pass


def train1(model, config):
    # random sampling with limits based on 

    # while steps < config.steps:
        # sample = get_sample()
        # model.batch_update_model(sample)
        # steps += config.batch_size * config.batch_length
        

        
        
    pass

def train2():
    # epoch style training
    # for e in epochs:
        # no. of batches = dataset / batch_size
        # for i in batch_num:
            # model.batch_update_model()

            # if logging_interval:
                # add tb plots
            
            # if video_logging_interval:
                # add video plots
    
    # shuffle dataset
    # 

    for _ in epochs:
        batch_count = dataset_size(config) / config.batch_size
        for _ in range(batch_count):
            model.batch_update_model()

    


def dataset_size(config):
    episode_size = 500
    num_episodes = len(config.datadir.glob('*.npz'))
    return (episode_size - config.batch_length) * num_episodes


def load_config(path):
    with open(path) as f:
        config = yaml.load(f)
    config = AttrDict(config)

    config.logdir = pathlib.Path('.')
    
    return config


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.config_file, args.override_config_file)
