import pathlib as P
import yaml
import torch

from torch.utils.tensorboard import SummaryWriter

from .management import *

class Logger():

    def __init__(self,
                 exp_name: str,
                 run_name: str,
                 configuration: dict,
                 verbose: bool = False,
                 log_gradients: bool = False,
                 log_data: bool = True,
                 checkpoint_frequency: int = 0,
                 
                 ):
        
        ### Logger Parameters ###
        self.verbose = verbose
        self.log_gradients = log_gradients
        self.log_data = log_data
        self.checkpoint_frequency = checkpoint_frequency

        ### Tensorboard Logger ###
        self.writer = SummaryWriter(str(self.run_dir/"logs"))

        ### Run Information ###
        self.exp_dir = P('exp_data') / exp_name
        self.run_dir = self.exp_dir / run_name
        self.config = configuration


        if self.verbose:
            output += f'-----Start Training of run {self.run_dir}-----\n'
            output += f'Logger Configuration:\n Log Gradients: {self.log_gradients}\n Log Data: {self.log_data}\n Checkpoint Frequency: {self.checkpoint_frequency}\n Verbose: {self.verbose}\n'
            output += f'Model Configuration:\n'
            output += yaml.dump(self.config, default_flow_style=False)
            print(output)

        self.init_run()

    def init_run(self):
        initiate_experiment(str(self.exp_dir))
        initiate_run(str(self.run_dir))

    def step(self,
            epoch: int,
            data: dict = None,
            model: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None,
            iteration: int = None):
        
        """
        Logger takes a step. The provided data is saved to tensorboard files if the logger is set up to save data.
        The logger automatically takes checkpoints in the frequency defined by the configuration.

        Arguments:
            Epoch (int): Current epoch.
            data (dict, optional): Data to log. The data should be provided through a dictionary of form:
                {
                'name': value,
                }
            model (torch.nn.Module, optional): Model to log.
            optimizer (torch.optim.Optimizer, optional): Optimizer to log.
            iteration (int, optional): Current iteration.
        
        
        """
        
        if data is not None:
            if self.verbose:
                output += f'-----Epoch {epoch} Results -----\n'
                output += yaml.dump(data, default_flow_style=False)
                print(output)
            if self.log_data:
                self.log(epoch, data, iteration)

        if model is not None and self.log_gradients:
            self.log_model_gradients(epoch, model, iteration)

        if self.save_frequency != 0 and (epoch % self.checkpoint_frequency == 0):

            self.checkpoint(epoch, model, optimizer)

    def log(self,
            epoch: int,
            data: dict,
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

        prefix_name = f'epoch_metrics/' if iteration is None else f'iteration_metrics/'

        for key, value in data.items():
            self.writer.add_scalar(prefix_name + key, int(value), (epoch if iteration is None else ((epoch-1)*self.config['num_iterations'] + iteration)))
    
    def log_model_gradients(self, 
                            epoch: int,
                            model: torch.nn.Module,
                            iteration: int = None):
        
        prefix_name = f'epoch_metrics/gradients/' if iteration is None else f'iteration_metrics/gradients/'

        for tag, value in model.named_parameters():
            if value.grad is not None:
                self.logger.add_histogram(prefix_name + tag + "/grad", value.grad.cpu(), epoch if iteration is None else ((epoch-1)*self.config['num_iterations'] + iteration))

    def checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):

        """
        Save state of the model and optionally of the optimizer to the checkpoint directory.
        
        """

        save_path = self.run_dir / "checkpoints" / f'checkpoint_{epoch}.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, str(save_path))

        if self.verbose:
            output += f'Checkpoint saved to: {save_path}\n' if optimizer is not None else f'Checkpoint (without optimizer) saved to: {save_path}\n'
            print(output)
    
    def save_config(self, save_path: str):
        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f)