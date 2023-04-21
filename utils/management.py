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

    exp_dir = P("exp_data") / exp_name

    if os.path.exists(str(exp_dir)):
        print("Experiment directory exists...")
        return -1

    os.makedirs(str(exp_dir))

    print('New experiment initialized to:' + str(exp_dir))

     
    return 1

def initiate_run(exp_name: str, config: dict = None, run_name: str = None):

    number_run = len(os.walk('dir_name').next()[1])

    run_dir = P("exp_data") / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))

    if not os.path.exists(str(P("exp_data") / exp_name)):
        print("Experiment directory not initialized yet")
        initiate_experiment(exp_name)

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

    run_dir = P("exp_data") / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))

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

def load_config(exp_name: str, run_name: str, config_name: str):
    config_path = P('config') / (config_name+'.yaml')
    run_path = P('exp_data') / exp_name / run_name
    try:
        with open(str(config_path), 'r') as f:
            config = yaml.safe_load(f)
    except:
        print("Config file does not exist")
        return -1
    try:
        with open(str(run_path / "config.yaml"), 'w') as f:
            yaml.safe_dump(config, f)
    except:
        print("Config file could not be saved to run directory")
        return -1
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_exp", action="store_true", help="Initialize experiment")
    parser.add_argument("--init_run", action="store_true", help="Initialize run")
    parser.add_argument("--clear_run", action="store_true", help="Clear run. Caution, potential data loss!")
    parser.add_argument("--conf", action='store_true' default=False, help="Load and save config file to run")
    parser.add_argument("-exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("-run_name", type=str, default=None help="run experiment")
    parser.add_argument("-config", type=str, default=None, help="Name of config file")
    parser.add_argument("-run_name", type=str, default=None)
    args = parser.parse_args()

    if args.init_exp:
        assert args.exp_name is not None
        initiate_experiment(args.exp_name)
    if args.init_run:
        assert args.exp_name is not None and args.run_name is not None
        initiate_run(args.exp_name, args.config, args.run_name)
    if args.clear_run:
        assert args.exp_name is not None and args.run_name is not None
        clear_run(args.exp_name, args.run_name)
    if args.conf:
        assert args.exp_name is not None and args.run_name is not None and args.config is not None
        load_config(args.exp_name, args.run_name, args.config)