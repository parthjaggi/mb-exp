import argparse

parser = argparse.ArgumentParser(description='Run model learning experiments')
parser.add_argument("config_file", type=str, default="configs/main.yaml", help='path of configuration file')
parser.add_argument("override_config_file", nargs='?', type=str, default=None, help='path of overriding configuration file')

# can be run in training or evaluation mode
# in eval you load the model, 
    # run over the loaded episodes
    # 