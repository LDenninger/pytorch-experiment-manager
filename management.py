import argparse
import os
import shutil

import yaml
from pathlib import Path as P

import config


"""
This file bundles all helper functions responsible for managing the experiments and related configurations.
"""


def initiate_experiment(exp_name):

    exp_dir = P("experiments") / exp_name

    if os.path.exists(str(exp_dir)):
        print("Experiment directory exists...")
        return -1

    os.makedirs(str(exp_dir))

    print('New experiment initialized to:' + str(exp_dir))

     
    return 1

def initiate_run(exp_name: str, config: dict = None, run_name: str = None):

    number_run = len(os.walk('dir_name').next()[1])

    run_dir = P("experiments") / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))

    if os.path.exists(str(run_dir)):
        print("Run directory already exists. Caution, existing files might be overwritten!")
        return 2
    try:
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "logs"))
        os.makedirs(str(run_dir / "checkpoints"))
        os.makedirs(str(run_dir / "plots"))
        os.makedirs(str(run_dir / "visualizations"))
    except Exception as e:
        print("Initialization failed: Directories could not be created")
        print(e)
        return -1
    if config is not None:
        try:
            with open(str(run_dir / "config.yaml"), 'w') as f:
                yaml.safe_dump(config, f,)
        except Exception as e:
            print("Initialization failed: Configuration file could not be created")
            print(e)
            return -1
    print('New run initialized to:' + str(run_dir))

    return 1

def clear_run(exp_name: str, run_name: str = None):

    run_dir = P("experiments") / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))

    if not os.path.exists(str(run_dir)):
        print("Run directory does not exist")
        return -1
    try:
        shutil.rmtree(str(run_dir / 'logs'))
        shutil.rmtree(str(run_dir / 'checkpoints'))
        shutil.rmtree(str(run_dir / 'plots'))
        shutil.rmtree(str(run_dir / 'visualizations'))
    except Exception as e:
        print("Reset of run failed: Directories could not be deleted")
        print(e)
        return -1

    return 1
