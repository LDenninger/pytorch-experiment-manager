import pathlib as P
import ruamel.yaml as yaml
import torch
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import optuna

import os
import copy

from .management import *

class Logger():

    def __init__(self,
                 exp_name: str = None,
                 run_name: str = None,
                 model_config: dict = None,
                 verbose: bool = False,
                 log_gradients: bool = False,
                 log_data: bool = True,
                 checkpoint_frequency: int = 0,
                 disabled: bool = False,
                 anon_mode: bool = False,
                 init_run: bool = False,
                 ):
        
        ### Logger Parameters ###
        self.verbose = verbose
        self.log_gradients = log_gradients
        self.log_data = log_data
        self.checkpoint_frequency = checkpoint_frequency

        self._internal_save = {}

        # The logger does not log at all 
        self.disabled = disabled

        # In anonymous mode, the logger does not write to disk but rather saves the data internally to be retrieved later from the user.
        self.anon_mode = anon_mode

        self.epoch = 1
        self.iteration = 1

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
        if self.anon_mode:
            print('Run environement not completely specified. Logger running in anonymous mode --> No logging to disk.')
        else:
            if "EXP_DIRECTORY" in os.environ:
                self.exp_dir = P(os.environ["EXP_DIRECTORY"]) / 'exp_data' / exp_name
            else:
                self.anon_mode = True
                print("Experiment directory not set. Please set the environment variable EXP_DIRECTORY.\n Logger running in anonymous mode --> No logging to disk.")
            self.exp_dir = P('exp_data') / exp_name
        self.run_dir = self.exp_dir / run_name
        

        self.model_config = model_config

        ### Tensorboard Logger ###
        self.writer = SummaryWriter(str(self.run_dir/"logs"))

        ### Initialize Data Solver ###



        if self.verbose:
            output = ''
            output += f' ------Start Training of run {self.run_name}-----\n\n'
            output += f' --Logger Configuration--\n  Log Gradients: {self.log_gradients}\n  Log Data: {self.log_data}\n  Checkpoint Frequency: {self.checkpoint_frequency}\n  Verbose: {self.verbose}\n'
            output += f'\n --Model Configuration--\n'
            for key, value in self.model_config.items():
                output += f'  {key}: {value}\n'
            print(output)
        if init_run:
            self.init_run()

    def init_run(self):
        initiate_experiment(self.exp_name)
        initiate_run(self.run_name)

    def step(self,
            epoch: int = None,
            data = None,
            data_resolved: bool = True,
            model: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None,
            iteration: int = None):
        
        """
        Logger takes a step. The provided data is saved to tensorboard files if the logger is set up to save data.
        The logger automatically takes checkpoints in the frequency defined by the configuration.

        Arguments:
            Epoch (int): Current epoch.
            data (optional): Data to log. The data should be provided through a dictionary of form:
            data_resolved (bool, optional): Whether or not the data is already resolved. 
                                            If the data is not resolved, the logger tries to resolve the data automatically. Defaults to True.
                Resolved format of data: 
                    {
                    'name': value,
                    }

            model (torch.nn.Module, optional): Model to log.
            optimizer (torch.optim.Optimizer, optional): Optimizer to log.
            iteration (int, optional): Current iteration.
        
        
        """

        ## Step forward to track training progress ##
        #Auto-step forward within logger 
        if self.iteration == self.model_config['num_iterations']:
            self.epoch += 1
            self.iteration = 0
        self.iteration += 1

        # Behavior when providing an epoch and iteration and decision on whether the step comes for epoch or iteration data
        if epoch is not None and iteration is not None:
            self.epoch = epoch+1
            self.iteration = iteration+1
        if epoch is not None and iteration is None:
            self.epoch = epoch+1
            self.iteration = 0
        elif epoch is None and iteration is not None:
            self.iteration = iteration+1

        if self.disabled:
            return 2

        try:
            if data is not None:
                if self.verbose:
                    output = ''
                    output += f'-----Epoch {epoch} Results -----\n'
                    output += yaml.dump(data, default_flow_style=False)
                    print(output)
                if self.log_data:
                    self.log(epoch, data, iteration)

            if model is not None and self.log_gradients:
                self.log_model_gradients(epoch, model, iteration)

            if self.checkpoint_frequency != 0 and (epoch % self.checkpoint_frequency == 0) and iteration is None:

                self.checkpoint(epoch, model, optimizer)
        except:
            print(f'Logging failed in Epoch: {self.epoch}, Iteration: {self.iteration}.')
            return -1
        
        return 1

    def log(self,
            epoch: int = None,
            data: dict = None,
            iteration: int = None
            ):
        """
        Log data provided through a dictionary or the model parameters to the local disk and/or tensorboard.

        Arguments:
            epoch (int): Current epoch.
            iteration (int, optional): Current iteration.
            data (dict, optional): Data to log. The data should be provided through a dictionary of form:
                {
                'name': value,
                }
       

        """

        if self.disabled:
            return 2

        prefix_name = f'epoch_metrics/' if iteration is None else f'iteration_metrics/'

        log_iter = self._get_log_iteration(epoch, iteration)

        for key, value in data.items():
            if value!=-1:
                self.writer.add_scalar(prefix_name + key, int(value), log_iter)
        return 1
    
    def log_model_gradients(self, 
                            epoch: int,
                            model: torch.nn.Module,
                            iteration: int = None):
        if self.disabled:
            return 2
        prefix_name = f'epoch_metrics/gradients/' if iteration is None else f'iteration_metrics/gradients/'

        for tag, value in model.named_parameters():
            if value.grad is not None:
                self.writer.add_histogram(prefix_name + tag + "/grad", value.grad.cpu(), epoch if iteration is None else ((epoch-1)*self.config['num_iterations'] + iteration))
        return 1

    def log_parameter_tuning(self, studies: list, trial_config: list, optimized_config: dict, name: str = None):

        if self.disabled:
            return 2
        

        save_dict = {}
        save_dict['base_config'] = self.model_config
        save_dict['final_config'] = optimized_config

        for i, study in enumerate(studies):

            s_config = trial_config[i]

            data_frame = study.trials_dataframe(attrs=('number', 'value', 'params'))
            data = data_frame[['number', 'value']+[('params_'+param['param'])for param in s_config['parameter']]].to_dict('records')

            best_params = {}

            opt_conf = copy.deepcopy(self.model_config)

            for key, value in study.best_params.items():
                best_params[key] = value
                opt_conf[key] = value
            
            save_dict[f'study_{i}'] = {
                        'trial_config': s_config,
                        'best_params': s_config,
                        'optimized_config': opt_conf,
                        'trial_data': data
            }

        self._anonymous_log(id='pOpt', data=save_dict)
        print('Hyperparameter tuning is getting logged internally...')

        if not self.anon_mode:
            print('Additionally saving logged to disk...')
            if name is None:
                name = '001'
                while os.path.exists(self.run_dir / "logs" / f'pOpt_{id}.yaml'):
                    name = str(int(id) + 1).zfill(3)

            save_path = self.run_dir / "logs" / f'pOpt_{name}.pth' if name is None else (self.run_dir / "checkpoints" / (name+'.pth'))
            
            yaml = yaml.YAML()
            yaml.dump(save_dict, save_path)

        
    
    def checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, name: str = None):

        """
            Save state of the model and optionally of the optimizer to the checkpoint directory.
        
        """
        if self.disabled:
            return 2
        save_path = self.run_dir / "checkpoints" / f'checkpoint_{epoch}.pth' if name is None else (self.run_dir / "checkpoints" / (name+'.pth'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, str(save_path))

        if self.verbose:
            output = ''
            output += f'Checkpoint saved to: {save_path}\n' if optimizer is not None else f'Checkpoint (without optimizer) saved to: {save_path}\n'
            print(output)
        return 1
    
    def save_config(self, save_path: str):
        yaml = yaml.YAML()
        yaml.dump(self.model_config, open(save_path, 'w'))

    def _anonymous_log(self, id: str, data):
        if self.disabled:
            return 2
        self._internal_save[id].append(data)
        return 1

    def _initialize_data_patterns(self):

        ### Data from training ###
        train_iter_loss_pattern = [self.model_config['num_iterations']]
        train_batch_iter_loss_pattern = [self.model_config['num_iterations'] , self.model_config['batch_size']]

    
    def _resolve_data(self, data):
        """
            Automatic resolve of the data into a dictionary of form: {'param': 'value'}

            TODO: Finish patterns and pattern matching.
        """

        def __resolve_data_structure(data, depth: int = 0):

            data_type = type(data)

            if data_type in [dict, list, tuple]:
                return {
                    'type': data_type,
                    'size': len(data),
                    'depth': depth,
                    'children': [__resolve_data_structure(item, depth+1) for item in data]
                }
            elif data_type in [str, int, float, bool]:
                return {
                    'type': data_type,
                    'value': data,
                }
            else:
                return {
                    'type': data_type,
                }
                

        data_type = type(data)

        # Resolve structure of nested dictionaries or lists
        
        if data_type in [dict, list, tuple]:
            data_resolve = copy.deepcopy(data)
            try:
                data_structure = __resolve_data_structure(data_resolve)
            except:
                print(f'Warning: Could not resolve logger data. Data was not logged. Please provide data in explicit form.')
                return None

        def __resolve(params: list, values: list):
            assert len(params) == len(values)
            for param, value in zip(params, values):
                data_resolved[param] = value
            

        data_resolved = {}

        ## Try resolve from evaluation data ##
        if ('evaluation' in self.model_config.keys()) and ('metrics' in self.model_config['evaluation'].keys()):
            # Value for each evaluation metric, 
            if (data_structure['size'] == self.model_config['evaluation']['metrics']['size']) and all(len(x)==2 for x in data_structure['children']):
                __resolve(self.model_config['evaluation']['metrics'], [x['value'] for x in data_structure['children']])
            # Value for each evaluation metric 

    def _get_log_iter(self, epoch: int = None, iteration: int = None):

        if epoch is not None and iteration is not None:
            n = epoch*self.model_config['num_iterations'] + iteration
        elif epoch is None and iteration is None:
            n =  self.epoch if self.iteration == 0 else self.epoch*self.model_config['num_iterations'] + self.iteration
        elif epoch is not None:
            n = epoch
        else:
            n = self.epoch*self.model_config['num_iterations'] + iteration
        
        return n