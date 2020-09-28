import torch
import torch.nn as nn

from trainer import Trainer
from trainer_utils import Result
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self):
        self.n_points = 40
        self.x = torch.randn(self.n_points, 10)
        self.y = torch.randn(self.n_points, 20)

    def __len__(self):
        return self.n_points

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DummyTrainer(Trainer):
    def __init__(self, n_epochs, batch_size, use_gpu=False,
                 checkpoint_dir=None, save_frequency=None, 
                 deterministic_seed=None, extended_save=False):

        super().__init__(n_epochs, batch_size, 
                         use_gpu=use_gpu,
                         checkpoint_dir=checkpoint_dir,
                         save_frequency=save_frequency, 
                         deterministic_seed=deterministic_seed,
                         extended_save=extended_save)

        self.model1 = nn.Linear(10, 20)
        self.model2 = nn.Linear(20,20)
        self.const = torch.ones(20)
        self.crit = nn.MSELoss()

        # visibility variables
        self.all_train = None
        self.all_valid = None

    def configure_optimizers(self):
        return torch.optim.SGD(list(self.model1.parameters()) + \
                               list(self.model2.parameters()), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y1 = self.model1(x) + self.const
        y2 = self.model2(y1)

        loss = self.crit(y2, y)

        res = Result(loss)
        res.abc = [torch.ones(2), torch.zeros(2)]
        return res

    def training_epoch_end(self, training_step_outputs):
        self.all_train = training_step_outputs

    def validation_step(self, batch, batch_idx):
        x = torch.ones(5, requires_grad=True)
        x = x*x
        return x

    def validation_epoch_end(self, validation_step_outputs):
        self.all_valid = validation_step_outputs
        return torch.mean(torch.cat(validation_step_outputs.loss))

    def testing_step(self, batch, batch_idx):
        pass

    def testing_epoch_end(self, testing_step_outputs):
        pass
