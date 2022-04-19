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
    arg_parser.add_argument("-d", "--dataset", help="Path to dataset", required=True)
    arg_parser.add_argument("-m", "--model", help="Path to model", required=(True if args.process == "test" else False))
    arg_parser.add_argument("-c", "--config", help="Path to model configuration", required=(True if args.process == "train" else False))

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    args = config_parser()

    with open(args.config, "r") as f:
        config = yaml.load(f)

    if args.process == "train":
        header("Training Configuration")
        print(config)

        vcnet = VCNet(config)

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
