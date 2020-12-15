import logging
import ray
import pprint
import yaml
import pathlib

from tools import AttrDict, num_of_sequence_samples, get_timestamp
from argparser import parser
from models.base_model import Model
from models.conv_model import ConvModel
from models.veh_model import VehModel



def run(config, override_config):
    config = load_config(config)

    if config.type in ['conv', 'conv_speed', 'class']:
        model = ConvModel(config)
    elif config.type == 'veh':
        model = VehModel(config)
    else:
        raise NotImplementedError

    model.train()

def train1(model, config):
    # random sampling from disk episodes
    steps = 0
        
    while steps < config.steps:
        model.batch_update_model()
        steps += config.batch_size * config.batch_length
        do_logging(model, config, steps)


def train2(model, config):
    # shuffled samples. epoch style training
    steps = 0
    
    for _ in config.epochs:
        batch_count = num_of_sequence_samples(config) / config.batch_size
        for _ in range(batch_count):
            model.batch_update_model()
            steps += config.batch_size * config.batch_length
            do_logging(model, config, steps)


def do_logging(model, config, steps):
    if model.should_eval(steps):
        # eval model on test episodes
        pass

    if model.should_log(steps):
        # do logging
        pass

    if model.should_image_log(steps):
        # save video
        pass
    
    if model.should_save_model(steps):
        # save model. qmix style: different for each save?
        # save current for now.
        model.save()
        pass


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    timestamp = get_timestamp()
    
    config.dir = pathlib.Path('.')
    config.resdir = pathlib.Path('.') / config.resdir
    config.logdir = config.resdir / timestamp
    config.datadir = config.dir / config.datadir

    config.resdir.mkdir(exist_ok=True)
    config.logdir.mkdir(exist_ok=True)
    return config


if __name__ == "__main__":
    args = parser.parse_args()
    run(args.config, args.override_config)
