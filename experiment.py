import torch

from trainer import Trainer
from datasets import CtDataset
from visualizer import Visualizer
from parser_options import parser
from unet_gen import UNetGenerator
from patch_disc import NLayerDiscriminator
from utils import get_group_dicts, ImagePool


class SimpleGAN(Trainer):
	def __init__(self, parsed_args, parsed_groups):
		super().__init__(**parsed_groups['trainer arguments'])

		self.gen = UNetGenerator(**parsed_groups['generator arguments'])
		self.disc = NLayerDiscriminator(**parsed_groups['discriminator arguments'])

		self.parsed_args = parsed_args
		self.vis = Visualizer()

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.gen.parameters(), lr=self.parsed_args.lr)
        opt2 = torch.optim.Adam(self.disc.parameters(), lr=self.parsed_args.lr)

        N = self.parsed_args.n_epochs
        N_start = int(N * self.parsed_args.frac_decay_start)
        sched_lamb = lambda x: 1.0 - max(0, x - N_start) / (N - N_start)

        sched1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda=sched_lamb)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=sched_lamb)

        return opt1, opt2, sched1, sched2


    def on_fit_start(self):
        self.vis.start()

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, training_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_outputs):
        pass


parser = Trainer.add_trainer_args(parser)
args = parser.parse_args()
group_dicts = get_group_dicts(args, parser)

train_dataset = CtDataset('train', **group_dicts['data arguments'])
valid_dataset = CtDataset('val', **group_dicts['data arguments'])

GAN_experiment = SimpleGAN(args, group_dicts)
GAN_experiment.fit(train_dataset, valid_dataset)
