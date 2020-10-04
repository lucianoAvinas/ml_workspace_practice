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
        self.un_normalize = lambda x: 255.*(1 + x.clamp(min=-1, max=1))/2.

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

    def _shared_step(self, batch, save_img, is_train):
        res = Result()

        X, Y_real = batch
        Y_fake = self.gen(X)

        if is_train:
            Y_pool = self.image_pool.query(Y_fake.detach())
        else:
            Y_pool = Y_fake
            res.recon_error = self.crit(Y_real, Y_fake)

        real_predict = self.disc(Y_real)
        fake_predict = self.disc(Y_pool)

        real_label = self.real_label.expand_as(real_predict)
        fake_label = self.fake_label.expand_as(fake_predict)

        disc_loss = 0.5 * (self.crit(real_predict, real_label) + \
                           self.crit(fake_predict, fake_label))

        if is_train:
            res.step(disc_loss)
        res.disc_loss = disc_loss

        gen_predict = self.disc(Y_fake)
        gen_loss = self.crit(gen_predict, real_label)

        if is_train:
            res.step(gen_loss)
        res.gen_loss = gen_loss

        if save_img:
            res.img = [self.un_normalize(X[:self.n_vis]),
                       self.un_normalize(Y_fake[:self.n_vis]),
                       self.un_normalize(Y_real[:self.n_vis])]
        return res

    def training_step(self, batch, batch_idx):
        res = self._shared_step(batch, 
                                save_img=(batch_idx == 0),
                                is_train=True)            
        return res

    def validation_step(self, batch, batch_idx):
        res = self._shared_step(batch, 
                                save_img=(batch_idx == self.sel_ind),
                                is_train=False)  
        return res

    def _shared_end(self, result_outputs, is_train):
        phase = 'Train' if is_train else 'Valid'

        self.vis.plot('Gen. Loss', phase + ' Loss', self.current_epoch, 
                       torch.mean(torch.stack(result_outputs.gen_loss)))
        self.vis.plot('Disc. Loss', phase + ' Loss', self.current_epoch, 
                       torch.mean(torch.stack(result_outputs.disc_loss)))

        collated_imgs = torch.cat([*torch.cat(result_outputs.img[0], dim=3)], dim=1)
        self.vis.show_image(phase + ' Images', collated_imgs)

    def training_epoch_end(self, training_outputs):
        self._shared_end(training_outputs, is_train=True)

    def validation_epoch_end(self, validation_outputs):
        self._shared_end(validation_outputs, is_train=False)
        self.sel_ind = random.randint(0, len(self.validation_loader) - 1)

        return torch.mean(torch.stack(validation_outputs.recon_error))


parser = Trainer.add_trainer_args(parser)
args = parser.parse_args()
group_dicts = get_group_dicts(args, parser)

train_dataset = CtDataset('train', **group_dicts['data arguments'])
valid_dataset = CtDataset('val', **group_dicts['data arguments'])

GAN_experiment = SimpleGAN(args, group_dicts)
GAN_experiment.fit(train_dataset, valid_dataset)
