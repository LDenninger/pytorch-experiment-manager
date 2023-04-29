import pathlib as P
import ruamel.yaml as yaml
import torch
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import optuna

import os
import copy
from collections import OrderedDict

from .management import *

class Logger():
    def __init__(self,     
                exp_name: str = None,
                    run_name: str = None,
                        model_config: dict = None,
                            writer: SummaryWriter = None,
                                save_external: bool = True,                                    
                                    save_internal: bool = False,                                    
                                ):
        
        ### Run Information ###
        if exp_name is not None and run_name is not None:
            self.exp_name = exp_name
            self.run_name = run_name
        elif exp_name is None and run_name is not None:
            self.run_name = run_name
            if 'ACTIVE_EXP' in os.environ:
                self.exp_name = os.environ['ACTIVE_EXP']
            else:
                self.run_name = None
                self.anon_mode = True
        elif exp_name is not None and run_name is None:
            self.exp_name = exp_name
            if 'ACTIVE_RUN' in os.environ:
                self.run_name = os.environ['ACTIVE_RUN']
            else:
                self.exp_name = None
                self.anon_mode = True
        elif exp_name is None and run_name is None:
            if 'ACTIVE_EXP' in os.environ and 'ACTIVE_RUN' in os.environ:
                self.exp_name = os.environ['ACTIVE_EXP']
                self.run_name = os.environ['ACTIVE_RUN']
            else:
                self.anon_mode = True
        else:
            self.exp_name = None
            self.run_name = None
            self.anon_mode = True
        if self.exp_name is not None and self.run_name is not None:
            self._init_run = True


        self.writer = SummaryWriter(str(self.run_dir/"logs")) if writer is None else writer

        self.model_config = model_config

        self.save_internal = save_internal
        self._internal_log = {}

        
        
    def log_data(self, epoch: int, data: dict,  iteration: int=None):
        """
            Log the data.
            Arguments:
                epoch (int): The current epoch. 
                data (dict): The data to log.
                    Format:
                            {
                                [name]: value,
                                ...
                            }
        
        """
        if self.save_external:
            prefix_name = f'epoch_metrics/' if iteration is None else f'iteration_metrics/'

            log_iter = epoch if iteration is None else (self.model_config['num_iterations']*(epoch-1) + iteration)

            for key, value in data.items():
                self.writer.add_scalar(prefix_name + key, int(value), log_iter)
            
        if self.save_internal:
            self._save_internal(data)
    
    def get_log(self):
        return self._internal_log

    def get_last_log(self):
        last_log = {}
        for key in self._internal_log.keys():
            last_log[key] = self._internal_log[key][-1]
        return last_log

    def enable_internal_log(self):
        self.save_internal = True
    
    def disable_internal_log(self):
        self.save_internal = False
    
    def _save_internal(self, data):
        for key, value in data.items():
            if not key in self._internal_log.keys():
                self._internal_log[key] = []
            self._internal_log[key].append(value)
    




class GradientInspector:
    """
        This class was adapted from https://github.com/angelvillar96/TemplaTorch
    
        Module that computes some statistics from the gradients of one parameter,
        and logs the stats into the Tensorboard
        Arguments:
            writer: TensorboardWriter
                TensorboardWriter object used to log into the Tensorboard
            layers: list of nn.Module
                Layers whose gradients are processed and logged into the Tensorboard
            names: list of strings
                Name given to each of the layers to track
            stats: list
            List with the stats to track. Possible stats are: ['Min', 'Max', 'Mean', 'Var', 'Norm']
    """

    STATS = ["Min", "Max", "Mean", "MeanAbs", "Var", "Norm"]
    FUNCS = {
        "Min": torch.min,
        "Max": torch.max,
        "Mean": torch.mean,
        "MeanAbs": lambda x: x.abs().mean(),
        "Var": torch.var,
        "Norm": torch.norm,
    }

    def __init__(self, writer, layers, names, stats=None):
        """ Module initializer """
        stats = stats if stats is not None else GradientInspector.STATS
        for stat in stats:
            assert stat in GradientInspector.STATS, f"{stat = } not included in {self.STATS = }"
        assert isinstance(layers, list), f"Layers is not list, but {type(layers)}..."
        assert len(layers) == len(names), f"{len(layers) = } and {len(names) = } must be the same..."
        for layer in layers:
            assert isinstance(layer, torch.nn.Module), f"Layer is not nn.Module, but {type(layer)}..."
            assert hasattr(layer, "weight"), "Layer does not have attribute 'weight'"

        self.writer = writer
        self.layers = layers
        self.names = names
        self.stats = stats

        print("Initializing Gradient-Inspector:")
        print(f"  --> Tracking stats {stats} of gradients in the following layers")
        for name, layer in zip(names, layers):
            print(f"    --> {name}: {layer}")
        return

    def __call__(self, step):
        """ Computing gradient stats and logging into Tensorboard """
        for layer, name in zip(self.layers, self.names):
            grad = layer.weight.grad
            for stat in self.stats:
                func = self.FUNCS[stat]
                self.writer.add_scalar(f"Grad Stats {name}/{stat} Grad", func(grad).item(), step)
        return
