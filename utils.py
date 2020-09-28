import os
import sys
import torch
import shutil
import warnings

from contextlib import contextmanager


def new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


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


@contextmanager
def suppress_stdout():
    #http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
