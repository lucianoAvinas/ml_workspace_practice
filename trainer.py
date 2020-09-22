import argparse
import torch
import torch.nn as nn

from abc import abstractmethod
from collections.abc import Iterable
from torch.utils.data import DataLoader
from utils import AbstactFinalMeta, runtimefinal, seed_everything, \
                  Result


class Trainer(metaclass=AbstactFinalMeta):
    def __init__(self, n_epochs, batch_size, n_cpus=0, use_gpu=False,
                 tune_cuddn=False, set_deterministic=None, pin_memory=False):
        self.n_epochs = n_epochs
        self.batch_size
        self.n_cpus = n_cpus
        self.use_gpu = use_gpu
        self.tune_cuddn = tune_cuddn
        self.set_deterministic = set_deterministic
        self.pin_memory = pin_memory

        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.current_epoch = 0

    def initialize_run(self):
        if self.set_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            seed_everything(self.set_deterministic)

        elif self.tune_cuddn:
            torch.backends.cudnn.benchmark = True


        si_units = [' ', 'k', 'M', 'G', 'T']
        print('-'*15+'Module Parameter Sizes'+'-'*15)
        obj_vars = vars(self)
        for var_name, var_value in obj_vars:
            if isinstance(var_value, nn.Module):

                var_nparams = str(sum(p.numel() for p in var_value.parameters()))
                unit_ind = min(len(var_nparams) // 3, len(si_units) - 1)
                print(f'{var_name} ({type(var_name).__name__}): '
                      f'{var_nparams[len(var_nparams) - 3*unit_ind:]}'
                      f'{si_units[unit_ind]}')

                var_value.to(self.device)
        print('-'*15+'-'*len('Module Parameter Sizes')+'-'*15)

    def update_trainer_parameters(self, new_loader_dict):
        common_keys = set(param_dict.keys()) & set(self.trainer_args.keys())

        for key in common_keys:
            self.trainer_args[key] = new_loader_dict[key]

    @staticmethod
    def add_trainer_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        trainer_args = parser.add_argument_group('trainer arguments')

        trainer_args.add_argument('--n_epochs', type=int, required=True,
                                  help='Number of epochs used for training.')
        trainer_args.add_argument('-batch', '--batch_size', type=int, required=True,
                                  help='DataLoader batch size.')
        trainer_args.add_argument('--n_cpus', type=int, default=0,
                                  help='Number of cpu used in DataLoader.')
        trainer_args.add_argument('--use_gpu', action='store_true',
                                  help='Sets PyTorch device to gpu.')
        trainer_args.add_argument('--tune_cuddn', action='store_true',
                                  help='Set cuddn benchmark flag to true.')
        trainer_args.add_argument('--use_gpu', action='store_true',
                                  help='Sets PyTorch device to gpu.')
        trainer_args.add_argument('--set_deterministic', type=int, default=None,
                                  help='Seed PyTorch and NumPy. Will overwrite tune_'
                                       'cudnn for deterministic alternative')
        trainer_args.add_argument('--pin_memory', action='store_true',
                                  help='Sets DataLoader pin_memory to true.')

        return parser

    @runtimefinal
    def fit(self, training_dataset, validation_dataset):
        # Instantiate Loaders
        # Setup optimizers
        # Setup visualizer
        # Loop # detach and aggregate (pre-fill list)
        # optim step and sched step
        # Checkpoint for timestep and track best val
        train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        valid_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        optimizers = self.configure_optimizers()
        optim_list, sched_list = dfs_detree_optimizer_list(optimizers)

        for _ in range(self.n_epochs):
            train_outputs = self.data_loop(train_loader, 'train', 
                                           self.training_step, optim_list)
            self.training_epoch_end(train_outputs)

            for sched in sched_list:
                sched_list.step()

            with torch.no_grad():
                valid_outputs = self.data_loop(valid_loader, 'eval', 
                                               self.validation_step)
                # checkpointing of the first arg epoch loss[0]
                self.validation_epoch_end(valid_outputs)

            self.current_epoch += 1


    @runtimefinal
    def test(self, testing_dataset, chckpt=None):
        # Instantiate Loader
        # If chkpt none use previously trained best
        test_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.n_cpus, 
                                 pin_memory=(self.use_gpu and self.pin_memory))

        with torch.no_grad():
            test_outputs = self.data_loop(test_loader, 'eval', self.testing_step)
            self.testing_epoch_end(test_outputs)

    @runtimefinal
    def data_loop(self, loader, phase, step_method, optim_list=[]):
        obj_vars = vars(self)
        for var_name, var_value in obj_vars:
            if isinstance(var_value, nn.Module):
                getattr(self, phase)()

        for i in range(len(optim_list)):
                optim_list[i].zero_grad()

        collected_results = [None] * len(loader)
        for batch_idx, batch in enumerate(loader):
            batch.to(self.device)
            step_result = Result(step_method(batch, batch_idx))

            for i in range(len(optim_list)):
                step_result.loss[i].backward()
                optim_list[i].step()
                optim_list[i].zero_grad()
            # This handles detach and to_cpu logic through __setattr__
            step_result.loss = step_result.loss
            #collect

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def training_epoch_end(self, training_step_outputs):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_epoch_end(self, validation_step_outputs):
        pass

    def testing_step(self, batch, batch_idx):
        raise NotImplementedError

    def testing_epoch_end(self, testing_step_outputs):
        raise NotImplementedError


def dfs_detree_optimizer_list(optimzers, max_depth=1):
    # Max depth set at 1 for PyTorchLightning compatibility
    optim_list = []
    sched_list = []

    def instance_check(subtree, curr_depth):
        if curr_depth < 0 or not isinstance(subtree, Iterable):
            raise TypeError('Node value must be of type',torch.optim.Optimizer.__name__, 
                            'or', torch.optim.lr_scheduler._LRScheduler.__name__)
        for subvals in subtree:
            if isinstance(subvals, torch.optim.Optimizer):
                optim_list.append(subvals)
            elif isinstance(subvals, torch.optim.lr_scheduler._LRScheduler):
                sched_list.append(subvals)
            else:
                instance_check(subvals, curr_depth - 1)

    if isinstance(optimizers, torch.optim.Optimizer):
        optim_list.append(optimizers)
    else:
        instance_check(optimizers, max_depth)

    if len(optim_list) == 0:
        raise ValueError('No optimizer has been set.')
    elif 1 < len(sched_list) != len(optim_list):
        raise ValueError('Scheduler and Optimizer lists cannot be broadcasted and have '
                         'unequal lengths.') # Length 1 sched_list is understood as being 
                                             # applied at once to all optimizers

    return optim_list, sched_list
