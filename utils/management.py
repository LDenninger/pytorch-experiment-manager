import argparse
import os
import shutil

import yaml
#import ruamel.yaml
#yaml = ruamel.yaml.YAML()
#yaml.default_flow_style = False
#yaml.indent(mapping=2, sequence=4, offset=2)
from pathlib import Path as P



"""
This file bundles all helper functions responsible for managing the experiments and related configurations.
"""


def initiate_experiment(exp_name):

    if "EXP_DIRECTORY" in os.environ:
        exp_dir = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name
    else:
        print("EXP_DIRECTORY environment variable not set. Using relative path...")
        exp_dir = P("exp_data") / exp_name


    if os.path.exists(str(exp_dir)):
        print("Experiment directory exists...")
        return -1

    os.makedirs(str(exp_dir))

    print('New experiment initialized to:' + str(exp_dir))

     
    return 1

def initiate_run(exp_name: str, config: dict = None, run_name: str = None):

    number_run = -1

    if "EXP_DIRECTORY" in os.environ:
        exp_dir = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name
        run_dir = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))
    else:
        print("EXP_DIRECTORY environment variable not set. Using relative path...")
        exp_dir = P("exp_data") / exp_name
        run_dir = P("exp_data") / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))

    if not os.path.exists(str(exp_dir)):
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
            yaml.dump(config, (run_dir / "config.yaml"))
        except Exception as e:
            print("Initialization failed: Configuration file could not be created")
            print(e)
            return -1
    print('New run initialized to:' + str(run_dir))

    return 1

def clear_run(exp_name: str, run_name: str = None):

    number_run = -1

    if "EXP_DIRECTORY" in os.environ:
        exp_dir = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name
        run_dir = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name / (run_name if run_name is not None else (f'run_{str(number_run + 1)}'))
    else:
        print("EXP_DIRECTORY environment variable not set. Using relative path...")
        exp_dir = P("exp_data") / exp_name
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

def clear_data():
    try:
        shutil.rmtree(str(P("exp_data")))
        os.makedirs(str(P("exp_data")))
    except Exception as e:
        print("Clear of data failed: Directories could not be deleted")
        print(e)
        return -1
    return 1

def write_config(exp_name: str, run_name: str, config_name: str):
    if "EXP_DIRECTORY" in os.environ:
        config_path = P(os.environ["EXP_DIRECTORY"]) / "config" / (config_name+'.yaml')
        run_path = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name / run_name
    else:
        print("EXP_DIRECTORY environment variable not set. Using relative path...")
        config_path = P("config") / (config_name+'.yaml')
        run_path = P("exp_data") / exp_name / run_name
    try:
        with open(str(config_path), 'r') as f:
            config = yaml.safe_load(f)
    except:
        print("Config file does not exist")
        return -1
    try:
        config['run_name'] = run_name
        with open(str(run_path / "config.yaml"), 'w') as f:
            yaml.safe_dump(config, f)
    except:
        print("Config file could not be saved to run directory")
        return -1
    return 1

def load_config(exp_name: str, run_name: str):
    if "EXP_DIRECTORY" in os.environ:
        config_path = P(os.environ["EXP_DIRECTORY"]) / "exp_data" / exp_name / run_name / 'config.yaml'
    else:
        print("EXP_DIRECTORY environment variable not set. Using relative path...")
        config_path = P("exp_data") / exp_name / run_name / 'config.yaml'
    try:
        with open(str(config_path), 'r') as f:
            config = yaml.safe_load(f)
    except:
        print("Config file does not exist")
        return -1
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_exp", action="store_true",default=False, help="Initialize experiment")
    parser.add_argument("--init_run", action="store_true",default=False, help="Initialize run")
    parser.add_argument("--clear_run", action="store_true",default=False, help="Clear run. Caution, potential data loss!")
    parser.add_argument("--conf", action='store_true', default=False, help="Load and save config file to run")
    parser.add_argument("--clear_data", action="store_true",default=False, help="Clear data")
    parser.add_argument("-exp", type=str, default=None, help="experiment name")
    parser.add_argument("-run", type=str, default=None, help="run experiment")
    parser.add_argument("-config", type=str, default=None, help="Name of config file")
    args = parser.parse_args()

    if args.exp is None and 'ACTIVATE_EXP' in os.environ:
        print("Current experiment: " + os.environ['ACTIVATE_EXP'])
        args.exp = os.environ['ACTIVATE_EXP']
    if args.run is None and 'ACTIVATE_RUN' in os.environ:
        print("Current run: " + os.environ['ACTIVATE_RUN'])
        args.run = os.environ['ACTIVATE_RUN']

    if args.init_exp:
        assert args.exp is not None
        initiate_experiment(args.exp)
    if args.init_run:
        assert args.exp is not None and args.run is not None
        initiate_run(args.exp, args.config, args.run)
    if args.clear_run:
        assert args.exp is not None and args.run is not None
        clear_run(args.exp, args.run)
    if args.conf:
        assert args.exp is not None and args.run is not None and args.config is not None
        write_config(args.exp, args.run, args.config)
    if args.clear_data:
        clear_data()