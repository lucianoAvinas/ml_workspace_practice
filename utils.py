import os
import torch
import random
import warnings
import numpy as np

from abc import ABCMeta
from collections import namedtuple


def runtimefinal(method):
    # Noctis Skytower - https://stackoverflow.com/a/39172073
    method.__final = True
    return method


def check_final(method):
    # Noctis Skytower - https://stackoverflow.com/a/39172073
    try:
        return method.__final
    except AttributeError:
        return False


class RuntimeFinalMeta(type):
    # Noctis Skytower - https://stackoverflow.com/a/39172073
    def __new__(mcs, name, bases, class_dict):
        final_methods = {key 
                         for base in bases
                         for key, value in vars(base).items()
                         if callable(value) and check_final(value)}
        
        conflicts = {key for key in class_dict if key in final_methods}
        if len(conflicts) > 0:
            raise RuntimeError(f'Can\'t instantiate final class {name} with '
                               'overwritten final methods '+', '.join(conflicts))
 
        return super().__new__(mcs, name, bases, class_dict)


AbstactFinalMeta = type('AbstactFinalMeta', (ABCMeta, RuntimeFinalMeta), {})


class Result(object):
    def __init__(self, loss, to_cpu=True):
        if isinstance(loss, Result):
            for key, val in vars(loss).items():
                super().__setattr__(key, val)
        else:
            super().__setattr__('to_cpu', to_cpu)
            super().__setattr__('_Result__loss', loss)

    def __setattr__(self, name, value):
        # Extra train_step dimension is added for collection process
        # Iterable values are not preemptively concatenated in case
        # individial elements are different dimensions.
        if name == 'loss':
            warnings.warn('\nLoss has been mangled as a private variable '
                          'and cannot be accessed externally. Tampering '
                          'with the mangled variable will affect Result\'s '
                          'detaching and collection process.\n')

        if isinstance(value, torch.Tensor):
            return super().__setattr__(name, [value.cpu().detach() if self.to_cpu
                                       else value.detach()])
        elif all(isinstance(subval, torch.Tensor) for subval in value):
            return super().__setattr__(name, [[subval.cpu().detach() if self.to_cpu 
                                       else subval.detach() for subval in value]])
        else:
            raise ValueError('Value set is not a torch.Tensor instance '
                             'or an iterable object of torch.Tensors')

    def detach_loss(self):
        self.__loss = self.__loss # calls __setattr__

    def get_loss(self):
        return self.__loss

    @staticmethod
    def collect(result_list):
        coll_dict = {}
        for result in result_list:
            var_dict = vars(result)

            for key in var_dict.keys():
                if key not in coll_dict:
                   coll_dict[key] = [] if key != 'to_cpu' else 0

                coll_dict[key] += var_dict[key]

        coll_dict.pop('to_cpu', None)
        coll_dict['loss'] = coll_dict.pop('_Result__loss')
        return namedtuple('CollectedResults', coll_dict.keys())(**coll_dict)


def get_group_dicts(args, parser):
    # https://stackoverflow.com/a/31520622
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title] = group_dict

    return arg_groups


def iterable_torch_eq(list1, list2):
    def iterate(sublist1, sublist2):
        if isinstance(sublist1, torch.Tensor) or isinstance(sublist2, torch.Tensor):
            if not (isinstance(sublist1, torch.Tensor) and isinstance(sublist2, torch.Tensor)):
                return False
            else:
                return torch.equal(sublist1, sublist2)

        elif len(sublist1) != len(sublist2):
            return False

        else:
            equality = 1
            for i in range(len(sublist1)):
                if equality == 0:
                    break
                equality *= iterate(sublist1[i], sublist2[i])
            return equality

    return iterate(list1, list2)


def seed_everything(seed=None):
    # PyTorchLightning Util
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        warnings.warn(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = random.randint(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
