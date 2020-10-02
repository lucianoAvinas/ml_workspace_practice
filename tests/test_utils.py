import torch
import unittest

from trainer_utils import Result
from utils import iterable_torch_eq


class TestResult(unittest.TestCase):
    def test_setattr(self):
        res = Result()

        res.abc = torch.ones(5,5,5)
        self.assertTrue(torch.equal(res.abc[0], torch.ones(5,5,5)))

        x = torch.tensor(3.14)
        res.abc = (x, x)
        self.assertTrue(iterable_torch_eq(res.abc, [(x, x)]))

        res.abc = (torch.ones(5).cuda(), torch.ones(5).cuda())
        self.assertFalse(res.abc[0][1].is_cuda)

    def test_step(self):
        m = torch.nn.Linear(5,5)
        x = torch.ones(5)
        opt1 = torch.optim.Adam(m.parameters(), lr=0.1)
        opt2 = torch.optim.Adam(m.parameters(), lr=0.1)
        Result.optim_list = [opt1, opt2]
        z = m.weight.data.clone()

        res = Result()
        y0 = m(x)
        y = torch.mean(y0*y0)
        res.step(y)
        res2 = Result()
        self.assertEqual(res.optim_phase, 1)
        self.assertEqual(res2.optim_phase, 1)
        self.assertFalse(torch.equal(m.weight.data, z))
        z = m.weight.data.clone()

        y0 = m(x)
        y = torch.mean(y0*y0)
        res.step(y)
        self.assertEqual(res.optim_phase, 0)
        self.assertEqual(res2.optim_phase, 0)
        self.assertFalse(torch.equal(m.weight.data, z))
        z = m.weight.data.clone()

        y0 = m(x)
        y = torch.mean(y0*y0)
        res.step(y, 0)
        self.assertEqual(res.optim_phase, 0)
        self.assertEqual(res2.optim_phase, 0)
        self.assertFalse(torch.equal(m.weight.data, z))

    def test_collect(self):
        res1 = Result()
        res1.a = torch.tensor(5.)
        res1.b = [torch.zeros(3,3)]
        res1.c = [torch.tensor(5.), torch.tensor(4.)]

        res2 = Result()
        res2.a = torch.tensor(5.)
        res2.b = torch.zeros(3,3)
        res2.c = [torch.tensor(5.), torch.tensor(4.)]

        res3 = Result()
        res3.f = torch.zeros(5)
        res3.ab = [torch.ones(4)]
        coll_res = Result.collect([res1, res2, res3])

        self.assertTrue(iterable_torch_eq(coll_res.a,
                                [torch.tensor(5.), torch.tensor(5.)]))
        self.assertTrue(iterable_torch_eq(coll_res.b,
                                [[torch.zeros(3,3)], torch.zeros(3,3)]))
        self.assertTrue(iterable_torch_eq(coll_res.c,
                                [[torch.tensor(5.), torch.tensor(4.)],
                                 [torch.tensor(5.), torch.tensor(4.)]]))
        self.assertTrue(iterable_torch_eq(coll_res.f, [torch.zeros(5)]))
        self.assertTrue(iterable_torch_eq(coll_res.ab, [[torch.ones(4)]]))

        coll_res = Result.collect([res1, None, res2])
        self.assertEqual(coll_res, None)
