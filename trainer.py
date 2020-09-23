import sys
import torch
import argparse
import torch.nn as nn

from abc import abstractmethod
from collections.abc import Iterable
from torch.utils.data import DataLoader
from utils import AbstactFinalMeta, runtimefinal, Result, seed_everything


class Trainer(metaclass=AbstactFinalMeta):
    def __init__(self, n_epochs, batch_size,n_cpus=0, use_gpu=False, checkpoint_dir=None,
                 save_frequency=None, tune_cuddn=False, set_deterministic=None, pin_memory=False):
        if save_frequency != None:
            assert save_frequency > 0

        self.n_epochs = n_epochs
        self.batch_size
        self.n_cpus = n_cpus
        self.use_gpu = use_gpu
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.tune_cuddn = tune_cuddn
        self.set_deterministic = set_deterministic
        self.pin_memory = pin_memory

        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.current_epoch = 0
        self.best_validation = None
        self.best_state = None
        self.nn_module_names = set([])

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
            if isinstance(var_value, torch.Tensor):
                setattr(self, var_name, var_value.to(self.device))

            elif isinstance(var_value, nn.Module):
                self.nn_module_names.add(var_name)

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
        trainer_args.add_argument('-chck_dir', '--checkpoint_dir', default=None,
                                  help='Directory to save and load checkpoints.')
        trainer_args.add_argument('-save_freq', '--save_frequency', type=int, default=None,
                                  help='Number of epochs between checkpoint saves.')
        trainer_args.add_argument('--set_deterministic', type=int, default=None,
                                  help='Seed PyTorch and NumPy. Will overwrite tune_'
                                       'cudnn for deterministic alternative')
        trainer_args.add_argument('--pin_memory', action='store_true',
                                  help='Sets DataLoader pin_memory to true.')

        return parser

    def compute_validation(self, validation_outputs):
        # To be overwritten for different validation calculation behavior
        if validation_outputs.loss[0].shape[0] > 1:
            # Take first entry of each losses tuple and average
            computed_validation = torch.mean(torch.stack(validation_outputs.loss))[0]
        else:
            computed_validation = torch.mean(torch.cat(validation_outputs.loss))

        return computed_validation

    def get_state_dicts(self):
        all_state_dicts = dict()
        for nn_module_name in self.nn_module_names:
            nn_module = getattr(self, nn_module_name)

            all_state_dicts[nn_module_name] = nn_module.cpu().state_dict()
            nn_module.to(self.device)

        return all_state_dicts

    def save_state(self):
        if self.checkpoint_dir == None:
            return None

        if on_epoch and self.save_frequency != None:
            if self.current_epoch % self.save_frequency == 0:
                all_state_dicts = self.get_state_dicts()
                for name, state in all_state_dicts.items():
                    torch.save(state, os.path.join(self.checkpoint_dir, name +  \
                                                   f'_epoch{self.current_epoch}.pth'))  

        if self.best_state != None:
            for name, state in self.best_state.items():
                torch.save(state, os.path.join(self.checkpoint_dir, name + f'_best.pth'))

    def load_state(state, chckpt_suffix):
        if chckpt_suffix:
            # standardize suffix
            chckpt_suffix = chckpt_suffix.split('.')[0] + '.pth'
            if chckpt_suffix[0] != '_':
                chckpt_suffix = '_' + chckpt_suffix

            if self.checkpoint_dir == None:
                raise Exception('Checkpoint directory must be set if checkpoint '
                                'suffix is specified')
        else:
            if self.best_state == None:
                raise Exception('No previous training and no checkpoint specified.')


        for nn_module_name in self.nn_module_names:
            if chckpt_suffix != None:
                module_path = os.path.join(self.checkpoint_dir, nn_module_name + \
                                           chckpt_suffix)
                state = torch.load(module_path)
            else:
                state = self.best_state[nn_module_name]
                
            nn_module = getattr(self, nn_module_name)
            nn_module.load_state_dict(state)
            nn_module.to(self.device)


    @runtimefinal
    def fit(self, training_dataset, validation_dataset):
        self.on_fit_start()
        self.initialize_run()

        train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        valid_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.n_cpus, 
                                  pin_memory=(self.use_gpu and self.pin_memory))

        optimizers = self.configure_optimizers()
        optim_list, sched_list = dfs_detree_optimizer_list(optimizers)

        self.best_validation = sys.float_info.max
        self.best_state = None
        leading_state = None
        for _ in range(self.n_epochs):
            train_outputs = self.data_loop(train_loader, 'train', 
                                           self.training_step, optim_list)
            self.training_epoch_end(train_outputs)

            for sched in sched_list:
                sched_list.step()

            with torch.no_grad():
                valid_outputs = self.data_loop(valid_loader, 'eval', 
                                               self.validation_step)

                curr_validation = self.compute_validation(valid_outputs)
                if curr_validation < self.best_validation:
                    self.best_validation = curr_validation
                    leading_state = self.get_state_dicts()
                            
                self.validation_epoch_end(valid_outputs)

            self.current_epoch += 1
            self.save_state()
        
        self.best_state = leading_state
        self.save_state()
        self.on_fit_end()

    @runtimefinal
    def test(self, testing_dataset, chckpt_suffix=None):
        self.on_test_start()
        self.initialize_run()
        self.load_state(chckpt_suffix)

        test_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.n_cpus, 
                                 pin_memory=(self.use_gpu and self.pin_memory))

        with torch.no_grad():
            test_outputs = self.data_loop(test_loader, 'eval', self.testing_step)
            self.testing_epoch_end(test_outputs)

        self.on_test_end()

    @runtimefinal
    def data_loop(self, loader, phase, step_method, optim_list=[]):
        for nn_module_name in self.nn_module_names:
            getattr(getattr(self, nn_module_name), phase)()

        for i in range(len(optim_list)):
                optim_list[i].zero_grad()

        all_results = [None] * len(loader)
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            step_result = Result(step_method(batch, batch_idx))

            for i in range(len(optim_list)):
                step_result.loss[i].backward()
                optim_list[i].step()
                optim_list[i].zero_grad()

            step_result.detach_loss()
            all_results[batch_idx] = step_result

        collected_results = Result.collect(all_results)
        return collected_results

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

    return optim_list, sched_list
