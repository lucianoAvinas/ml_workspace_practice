import os
import sys
import torch
import shutil
import random
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


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/util/image_pool.py
class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, replay_prob):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        self.replay_prob = replay_prob
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > self.replay_prob:  # by p% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by (1-p)% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


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
