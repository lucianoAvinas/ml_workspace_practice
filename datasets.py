import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class CtDataset(Dataset):
    def __init__(self, train_phase, data_dir, image_size, max_ct_value, reverse_direction):
        self.data_loc = os.path.join(data_dir, str(image_size), train_phase)
        
        self.img_types = ['A', 'B'] if reverse_direction else ['B', 'A']

        self.X_nums = sorted([s.split('_')[0] for s in os.listdir(
                              self.data_loc+self.img_types[0])], key=lambda x: int(x))
        self.Y_nums = sorted([s.split('_')[0] for s in os.listdir(
                              self.data_loc+self.img_types[1])], key=lambda x: int(x))

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(
                                                lambda x: x.float()/max_ct_value),
                                             transforms.Normalize((0.5,), (0.5,))
                                             ])
        # Don't forget to un-normalize to [0, 1] when displaying images

    def __len__(self):
        return len(self.X_nums)

    def __getitem__(self, index):
        X_file = os.path.join(self.data_loc+self.img_types[0],
                              self.X_nums[index]+'_'+self.img_types[0]+'.tiff')
        Y_file = os.path.join(self.data_loc+self.img_types[1],
                              self.Y_nums[index]+'_'+self.img_types[1]+'.tiff')

        X = self.transform(Image.open(X_file))
        Y = self.transform(Image.open(Y_file))
        return X, Y