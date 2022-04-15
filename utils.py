# Utility and helper functions
import os
import yaml

def header(text):
    print("-"*80)
    print(f"\t\t\t\t{head.upper()}")
    print("-"*80)


def generate_model_name(config):
    model_name = f"{config['model']}_{config['dataset']}_{config['batch_size']}_{config['lr']}_{config['epochs']}_{config['seed']}"
    return model_name

