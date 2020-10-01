import os
import torch
import random
import inspect
import warnings
import numpy as np

from abc import ABCMeta
from collections import namedtuple, Iterable


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
    optim_phase = 0
    optim_list = None

    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            return object.__setattr__(self, name, [value.cpu().detach()])

        elif all(isinstance(subval, torch.Tensor) for subval in value):
            return object.__setattr__(self, name, [[subval.cpu().detach() 
                                                    for subval in value]])
        else:
            return object.__setattr__(self, name, value)

    @classmethod
    def reset_phase(cls):
        cls.optim_phase = 0

    @classmethod
    def step(cls, loss, sel_ind=None):
        if sel_ind == None:
            sel_ind = cls.optim_phase
            incr = 1
        else:
            incr = 0

        sel_optim = cls.optim_list[sel_ind]

        sel_optim.zero_grad()
        loss.backward()
        sel_optim.step()

        cls.optim_phase = (cls.optim_phase + incr) % len(cls.optim_list)

    @classmethod
    def collect(cls, result_list):
        coll_dict = {}  
        for result in result_list:
            var_dict = vars(result)

            for key in var_dict.keys():
                if key not in coll_dict:
                    coll_dict[key] = []

                coll_dict[key] += var_dict[key]

        return namedtuple('CollectedResults', coll_dict.keys())(**coll_dict)


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
