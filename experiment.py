import torch
import random

from trainer import Trainer
from datasets import CtDataset
from trainer_utils import Result
from visualizer import Visualizer
from parser_options import parser
from unet_gen import UNetGenerator
from patch_disc import NLayerDiscriminator
from utils import get_group_dicts, ImagePool, init_weights


class SimpleGAN(Trainer):
    def __init__(self, parsed_args, parsed_groups):
        super().__init__(**parsed_groups['trainer arguments'])

        self.gen = UNetGenerator(**parsed_groups['generator arguments'])
        self.disc = NLayerDiscriminator(**parsed_groups['discriminator arguments'])

        init_weights(self.gen)
        init_weights(self.disc)

        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)

        self.crit = torch.nn.MSELoss() # LSGAN
        self.image_pool = ImagePool(parsed_args.pool_size, parsed_args.replay_prob)

        self.sel_ind = 0
        self.un_normalize = lambda x: (1 + x.clamp(min=-1, max=1))/2.

        self.parsed_args = parsed_args
        self.n_vis = parsed_args.n_vis
        self.vis = Visualizer()
        
    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.disc.parameters(), lr=self.parsed_args.lr)
        opt2 = torch.optim.Adam(self.gen.parameters(), lr=self.parsed_args.lr)

        N = self.parsed_args.n_epochs
        N_start = int(N * self.parsed_args.frac_decay_start)
        sched_lamb = lambda x: 1.0 - max(0, x - N_start) / (N - N_start)

        sched1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda=sched_lamb)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=sched_lamb)

        return opt1, opt2, sched1, sched2

    def on_fit_start(self):
        self.vis.start()

    def on_fit_end(self):
        self.vis.stop()

    def _shared_step(X, Y_fake, Y_real, is_train=False):
        res = Result()

        if is_train:
            Y_pool = self.image_pool(Y_fake.detach())
        else:
            Y_pool = Y_fake

        real_label = self.real_label.expand(X.size()[0])
        fake_label = self.fake_label.expand(X.size()[0])

        disc_loss = 0.5 * (self.crit(self.disc(Y_real), real_label) + \
                           self.crit(self.disc(Y_pool), fake_label))

        if is_train:
            res.step(disc_loss)
        res.disc_loss = disc_loss

        disc_class = self.disc(Y_fake)
        gen_loss = self.crit(disc_class, real_label)

        if is_train:
            res.step(gen_loss)
        res.gen_loss = gen_loss

        return res

    def training_step(self, batch, batch_idx):
        X, Y_real = batch
        Y_fake = self.gen(X)
        
        res = self._shared_step(X, Y_fake, Y_real)

        if batch_idx == 0:
            res.img = [self.un_normalize(X[:self.n_vis]),
                       self.un_normalize(Y_fake[:self.n_vis]),
                       self.un_normalize(Y_real[:self.n_vis])]

        return res

    def training_epoch_end(self, training_outputs):
        #loss = torch.mean(training_step_outputs.loss) # break into gen and disc
        collated = torch.cat(training_outputs.img[0], dim=3)
        #self.vis.show_image('Train Images', collated_imgs)

    def validation_step(self, batch, batch_idx):
        X, Y_real = batch
        Y_fake = self.gen(X)

        res = self._shared_step(X, Y_fake, Y_real, is_eval=True)
        res.recon_error = self.crit(Y_real, Y_fake)

        if batch_idx == self.sel_ind:
            res.img = [self.un_normalize(X[:self.n_vis]),
                       self.un_normalize(Y_fake[:self.n_vis]),
                       self.un_normalize(Y_real[:self.n_vis])]

        return res

    def validation_epoch_end(self, validation_outputs):
        collated = torch.cat(validation_outputs.img[0], dim=3)
        self.sel_ind = random.randint(0, 
                              len(self.validation_dataset) - 1)
        # return recon error


parser = Trainer.add_trainer_args(parser)
args = parser.parse_args()
group_dicts = get_group_dicts(args, parser)

train_dataset = CtDataset('train', **group_dicts['data arguments'])
valid_dataset = CtDataset('val', **group_dicts['data arguments'])

GAN_experiment = SimpleGAN(args, group_dicts)
GAN_experiment.fit(train_dataset, valid_dataset)
