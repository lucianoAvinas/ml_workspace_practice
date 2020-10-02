import os
import torch
import shutil
import unittest

from utils import suppress_stdout
from trainer import Trainer, dfs_detree_optimizer_list
from tests.toy_experiment import DummyDataset, DummyTrainer


class TestTrainer(unittest.TestCase):
    def test_dfs_detree_optimizer(self):
        opt = torch.optim.SGD([torch.nn.Parameter()], 0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1)

        opt_list, sched_list = dfs_detree_optimizer_list(opt)
        self.assertEqual(opt_list[0], opt)

        opt_list, sched_list = dfs_detree_optimizer_list((opt, sched))
        self.assertEqual(opt_list[0], opt)
        self.assertEqual(sched_list[0], sched)

        opt_list, sched_list = dfs_detree_optimizer_list(([opt, opt], sched))
        self.assertEqual(opt_list[0], opt)
        self.assertEqual(opt_list[1], opt)
        self.assertEqual(sched_list[0], sched)

        opt_list, sched_list = dfs_detree_optimizer_list((opt, sched, opt, sched))
        self.assertEqual(opt_list[0], opt)
        self.assertEqual(opt_list[1], opt)
        self.assertEqual(sched_list[0], sched)
        self.assertEqual(sched_list[1], sched)

    def test_update_trainer_params(self):
        dt = DummyTrainer(0, 2)
        update = {'use_gpu': True, 'abc': 5}
        dt.update_trainer_params(update)
        self.assertTrue(dt.use_gpu)
        self.assertEqual(getattr(dt, 'abc', None), None)

    def test_initialize_run(self):
        dt = DummyTrainer(0, 2, use_gpu=True, deterministic_seed=50)

        self.assertTrue(next(dt.model1.parameters()).is_cuda)
        self.assertTrue(next(dt.model2.parameters()).is_cuda)
        self.assertTrue(dt.const.is_cuda)

        dt = DummyTrainer(0, 2)
        dt.a = torch.nn.Linear(512, 512)
        dt.b = torch.nn.Sequential(torch.nn.Linear(512, 1024), 
                                  torch.nn.Linear(1024, 1024))
        print(dt)

    def test_get_state_dict(self):
        dt = DummyTrainer(0, 2, use_gpu=True)
        dt.param = torch.nn.Parameter(torch.ones(5))

        all_states = dt.get_state_dicts()
        self.assertFalse(any(val['weight'].is_cuda 
                         for val in all_states.values()))
        self.assertNotIn('const', all_states)
        self.assertNotIn('param', all_states)
        self.assertTrue(all(getattr(dt, key).weight.is_cuda 
                            for key in all_states.keys()))
    
    def test_save_state(self):
        dt = DummyTrainer(1, 2, checkpoint_dir='tests/ex', 
                          save_frequency=1)
        dd1, dd2 = DummyDataset(), DummyDataset()
        with suppress_stdout():
            dt.fit(dd1, dd2)
        dir_w_pths = os.listdir('tests/ex/epoch_1')
        self.assertIn('model1_epoch_1.pth', dir_w_pths)
        self.assertIn('model2_epoch_1.pth', dir_w_pths)
        self.assertNotIn('const_epoch_1.pth', dir_w_pths)
        self.assertNotIn('crit_epoch_1.pth', dir_w_pths)

        dir_w_pths = os.listdir('tests/ex/best_state')
        self.assertIn('model1_best_state.pth', dir_w_pths)
        self.assertIn('model2_best_state.pth', dir_w_pths)
        self.assertNotIn('const_best_state.pth', dir_w_pths)
        self.assertNotIn('crit_best_state.pth', dir_w_pths)

        shutil.rmtree('tests/ex')

    def test_load_state(self):
        dt =  DummyTrainer(6, 2)
        dd1, dd2 = DummyDataset(), DummyDataset()
        with suppress_stdout():
            dt.fit(dd1, dd2)
        dt.test(dd1)

        dt =  DummyTrainer(6, 2, checkpoint_dir='tests/ex',
                           save_frequency=3)
        with suppress_stdout():
            dt.fit(dd1, dd2)
        dt.test(dd1, 'epoch_3')

        shutil.rmtree('tests/ex')

    def test_data_loop(self):
        dd1, dd2 = DummyDataset(), DummyDataset()
        dt1 = DummyTrainer(6, 2, deterministic_seed=50)
        dt2 = DummyTrainer(6, 2, deterministic_seed=50)

        self.assertTrue(torch.equal(dt1.model1.weight.data, 
                                    dt2.model1.weight.data))
        self.assertTrue(torch.equal(dt1.model2.weight.data, 
                                    dt2.model2.weight.data))
        self.assertTrue(torch.equal(dt1.const.data, 
                                    dt2.const.data))

        dt1 = DummyTrainer(6, 2, deterministic_seed=50)
        with suppress_stdout():
            dt1.fit(dd1, dd2)

        dt2 = DummyTrainer(6, 2, deterministic_seed=50)
        with suppress_stdout():
            dt2.fit(dd1, dd2)

        self.assertTrue(torch.equal(dt1.model1.weight.data, 
                                    dt2.model1.weight.data))
        self.assertTrue(torch.equal(dt1.model2.weight.data, 
                                    dt2.model2.weight.data))
        self.assertTrue(torch.equal(dt1.const.data, 
                                    dt2.const.data))

