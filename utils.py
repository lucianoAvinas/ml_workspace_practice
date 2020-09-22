import os
import torch
import random
import numpy as np

from abc import ABCMeta


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
            for key, val in vars(loss):
                super().__setattr__(key, val)
        else:
            super().__setattr__('to_cpu', to_cpu)
            super().__setattr__('loss', loss)

    # Only takes tensors or an iterable object of tensors (not dicts)
    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            super().__setattr__(name,
                torch.Tensor([value.cpu().detach() if self.to_cpu 
                              else value.detach()]))
        else:
            super().__setattr__(name,
                torch.cat([subval.cpu().detach() if self.to_cpu
                           else subval.detach() for subval in value]))

    @staticmethod
    def collect(result_list):



def get_group_dicts(args, parser):
    # https://stackoverflow.com/a/31520622
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title] = group_dict

    return arg_groups


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
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
