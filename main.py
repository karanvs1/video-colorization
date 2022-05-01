# Main
import argparse
import os
import yaml

from train import *
from test import *
from utils import *


def config_parser():
    arg_parser = argparse.ArgumentParser(description="A simple script to convert a .csv file to a .json file.")

    arg_parser.add_argument("-p", "--process", help="train | test", required=True)
    arg_parser.add_argument("-m", "--model", help="Path to model", required=False)  # (True if arg.process == "test" else False))
    arg_parser.add_argument("-c", "--config", help="Path to configuration", required=True)  # (True if args.process == "train" else False))

    args = arg_parser.parse_args()

    return args


"""
Sample Run commands
Training : python3 main.py -p train -c 'test_config.yaml'
Testing : python3 main.py -p test -m 'saved_models/<model_name>' -c 'inference_config.yaml'

"""


if __name__ == "__main__":
    args = config_parser()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    verify_config(config)

    if args.process == "train":
        header("Training Configuration")
        print(config)

        header("Training")
        vcnet = VCNetSetup(config)

        vcnet.prepare()
        vcnet.train()
        vcnet.save()

    elif args.process == "test":
        header("Testing Configuration")
        print(args.config)

        colorizer = VideoColorizer(config)

        colorizer.prepare()
        colorizer.test()
        colorizer.save_results()

    elif args.process == "predict":
        header("Running prediction")
        colorizer = VideoColorizer(config)

        colorizer.prepare()
        colorizer.predict()
