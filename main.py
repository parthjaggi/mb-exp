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


def train(model):
    # for e in epochs:
        # no. of batches = dataset / batch_size
        # for i in batch_num:
            # model.batch_update_model()

            # if logging_interval:
                # add tb plots
            
            # if video_logging_interval:
                # add video plots
    pass


def load_config(path):
    with open(path) as f:
        config = yaml.load(f)
    config = AttrDict(config)

    config.logdir = pathlib.Path('.')
    
    return config


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.config_file, args.override_config_file)
