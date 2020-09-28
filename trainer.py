import os
import sys
import torch
import inspect
import warnings
import argparse
import torch.nn as nn

from abc import abstractmethod
from utils import new_folder
from collections.abc import Iterable
from torch.utils.data import DataLoader
from trainer_utils import AbstactFinalMeta, runtimefinal, Result, seed_everything


class Trainer(metaclass=AbstactFinalMeta):
    def __init__(self, n_epochs, batch_size, n_cpus=0, use_gpu=False, checkpoint_dir=None,
                 save_frequency=None, tune_cuddn=False, deterministic_seed=None, 
                 pin_memory=False, extended_save=False):
        if save_frequency != None:
            assert save_frequency > 0

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.use_gpu = use_gpu
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.tune_cuddn = tune_cuddn
        self.deterministic_seed = deterministic_seed
        self.pin_memory = pin_memory
        self.extended_save = extended_save

        self.initialize_trainer()

        self.current_epoch = 0
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.optim_mapping = None

        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None

        self.best_validation = None
        self.best_state = None

        self.optim_list = None
        self.sched_list = None

    def initialize_trainer(self):
        if self.deterministic_seed != None:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            seed_everything(self.deterministic_seed)

        elif self.tune_cuddn:
            torch.backends.cudnn.benchmark = True

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module) or isinstance(value, torch.Tensor):
            # does not add module or parameter to optimizer(s)
            # its better practice handle all dynamic growth within a module
            # and assign this module parameters in configure_optimizer
            object.__setattr__(self, name, value.to(self.device))
        else:
            object.__setattr__(self, name, value)

    def __str__(self):
        coll_str = ''
        si_units = [' ', 'k', 'M', 'G', 'T']

        coll_str += '-'*10+'Module Parameter Sizes'+'-'*10+'\n'
        for name, submodule in vars(self).items():
            if isinstance(submodule, nn.Module):
                n_params = count_params(submodule) 
                if n_params == 0:
                    continue

                n_params = str(n_params)
                unit_ind = min((len(n_params) - 1)//3, len(si_units) - 1)

                coll_str += f'{name} ({type(submodule).__name__}): ' + \
                            f'{n_params[:len(n_params) - 3*unit_ind]}' + \
                            f'{si_units[unit_ind]}\n'

        coll_str += '-'*10+'-'*len('Module Parameter Sizes')+'-'*10+'\n'
        return coll_str

    def update_trainer_params(self, new_loader_dict):
        trainer_args = inspect.signature(Trainer).parameters

        for key in trainer_args.keys():
            if key in new_loader_dict:
                setattr(self, key, new_loader_dict[key])

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
        trainer_args.add_argument('-chck_dir', '--checkpoint_dir', default=None,
                                  help='Directory to save and load checkpoints.')
        trainer_args.add_argument('-save_freq', '--save_frequency', type=int, default=None,
                                  help='Number of epochs between checkpoint saves.')
        trainer_args.add_argument('--deterministic_seed', type=int, default=None,
                                  help='Seed PyTorch and NumPy. Will overwrite tune_'
                                       'cudnn for deterministic alternative')
        trainer_args.add_argument('--pin_memory', action='store_true',
                                  help='Sets DataLoader pin_memory to true.')
        trainer_args.add_argument('--extended_save', action='store_true',
                                  help='Includes all Trainer instance variables that '
                                       'are torch.tensor objects in the Trainer save '
                                       'operation. This includes torch.nn.Parameter '
                                       'objects.')
        return parser

    def get_state_dicts(self):
        all_state_dicts = dict()
        for name, submodule in vars(self).items():
            if isinstance(submodule, nn.Module) and count_params(submodule) > 0:
                all_state_dicts[name] = submodule.cpu().state_dict()
                submodule.to(self.device)

            elif self.extended_save and isinstance(submodule, torch.Tensor):
                all_state_dicts[name] = submodule.cpu()

        return all_state_dicts

    def save_state(self):
        if self.checkpoint_dir == None:
            return None

        if self.save_frequency != None and (self.current_epoch + 1) % self.save_frequency == 0:
            epochs_passed = self.current_epoch + 1
            all_state_dicts = self.get_state_dicts()
            new_folder(os.path.join(self.checkpoint_dir, f'epoch_{epochs_passed}'))

            for name, state in all_state_dicts.items():
                torch.save(state, os.path.join(self.checkpoint_dir, f'epoch_{epochs_passed}',
                                               name + f'_epoch_{epochs_passed}.pth'))
        if self.best_state != None:
            new_folder(os.path.join(self.checkpoint_dir, 'best_state'))

            for name, state in self.best_state.items():
                torch.save(state, os.path.join(self.checkpoint_dir, 'best_state',
                                               name + f'_best_state.pth'))

    def load_state(self, chckpt_suffix):
        # Either loads previously trained best state or specified folder location
        # In its current implementation, load_state derives the folder location 
        # from the chckpt_suffix used.

        if chckpt_suffix != None:
            if self.checkpoint_dir == None:
                raise Exception('Checkpoint directory must be set if checkpoint '
                                'suffix is specified')
            # standardize suffix
            chckpt_suffix = chckpt_suffix.split('.')[0] + '.pth'
            if chckpt_suffix[0] != '_':
                chckpt_suffix = '_' + chckpt_suffix

            load_dir = os.path.join(self.checkpoint_dir, chckpt_suffix.split('.')[0][1:])
            state_generator = lambda submod_name: torch.load(os.path.join(load_dir, 
                                                                          submod_name))
            submod_pths = os.listdir(load_dir)
            pth_to_name = lambda pth_name: pth_name.split('_')[0]
        else:
            if self.best_state == None:
                raise Exception('No previous training and no checkpoint specified.')

            state_generator = lambda submod_name: self.best_state[submod_name]
            submod_pths = self.best_state.keys()
            pth_to_name = lambda pth_name: pth_name
     
        for submod_pth in submod_pths:
            state = state_generator(submod_pth)
            submodule_name = pth_to_name(submod_pth)
  
            submodule = getattr(self, submodule_name, None)
            if submodule == None or isinstance(submodule, torch.Tensor):
                setattr(self, submodule_name, state.to(self.device))

            elif isinstance(submodule, nn.Module):
                submodule.load_state_dict(state)
                submodule.to(self.device)

    @runtimefinal
    def fit(self, training_dataset, validation_dataset):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        valid_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        optimizers = self.configure_optimizers()
        self.optim_list, self.sched_list = dfs_detree_optimizer_list(optimizers)

        self.best_validation = sys.float_info.max
        self.best_state = None
        leading_state = None

        print(self)

        self.on_fit_start()
        for self.current_epoch in range(self.n_epochs):
            train_outputs = self.data_loop(train_loader, 'train', 
                                           self.training_step)
            self.training_epoch_end(train_outputs)

            for scheduler in self.sched_list:
                scheduler.step()

            with torch.no_grad():
                valid_outputs = self.data_loop(valid_loader, 'eval', 
                                               self.validation_step)
                computed_valid = self.validation_epoch_end(valid_outputs)

                if isinstance(computed_valid, torch.Tensor):
                    computed_valid = computed_valid.item()

                if computed_valid == None:
                    warnings.warn('\n If no aggregate loss is returned by '
                                  'validation_epoch_end, then Trainer can\'t '
                                  'keep track of best validation state.')

                elif computed_valid < self.best_validation:
                    self.best_validation = computed_valid 
                    leading_state = self.get_state_dicts()

            self.save_state()
        
        self.best_state = leading_state
        self.save_state()
        self.on_fit_end()

    @runtimefinal
    def test(self, testing_dataset, chckpt_suffix=None):
        self.load_state(chckpt_suffix)

        self.testing_dataset = testing_dataset

        test_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.n_cpus, 
                                 pin_memory=(self.use_gpu and self.pin_memory))
        self.on_test_start()
        with torch.no_grad():
            test_outputs = self.data_loop(test_loader, 'eval', self.testing_step)
            self.testing_epoch_end(test_outputs)

        self.on_test_end()

    @runtimefinal
    def data_loop(self, loader, phase, step_method):
        for submodule in vars(self).values():
            if isinstance(submodule, nn.Module):
                getattr(submodule, phase)()

        all_results = [None] * len(loader)
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)

            elif type(batch) == list or type(batch) == tuple:
                for i in range(len(batch)):
                    batch[i] = batch[i].to(self.device)

            elif type(batch) == dict:
                for key,val in batch.items():
                    batch[key] = val.to(self.device)

            else:
                raise ValueError('Compatible Datasets have output elements of '
                                 'type: torch.Tensor, list, tuple, or dict. All'
                                 'collection compatible types must have '
                                 'torch.Tensor items.')

            step_result = Result(step_method(batch, batch_idx))

            if phase == 'train':
                step_loss = step_result.get_loss()
                optim_mapping = self.optim_index_mapping()
                
                for i in range(len(step_loss)):
                    optim_ind = optim_mapping(i)

                    self.optim_list[optim_ind].zero_grad()
                    step_loss[i].backward()
                    self.optim_list[optim_ind].step()

            step_result.detach_loss()
            all_results[batch_idx] = step_result

        collected_results = Result.collect(all_results)
        return collected_results

    def optim_index_mapping(self):
        # If number of losses known beforehand
        # this can be changed appropriately.

        # For out of the box use
        # Ex: train_step returns D_loss, G_loss
        # return optimizers in dfs order D_optim, G_optim

        return lambda x: min(x, len(self.optim_list)-1)

    @staticmethod
    def join_parameters(lst_of_modules):
        lst_of_parameters = []
        for module in lst_of_modules:
            if isinstance(module, nn.Module):
                lst_of_parameters += list(module.parameters())
            elif isinstance(module, nn.Parameter):
                lst_of_parameters += [module]
            else:
                raise ValueError('list must be composed of torch.nn.Module '
                                 'objects and torch.nn.Parameter objects.')
        return lst_of_parameters

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

    # User defined start/end routines
    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass


def count_params(nn_module):
    return sum(p.numel() for p in nn_module.parameters()) 


def dfs_detree_optimizer_list(optimizers, max_depth=1):
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

    return optim_list, sched_list
