import torch
import unittest

from utils import Result, iterable_torch_eq


class TestResult(unittest.TestCase):
    def test_init(self):
        x = torch.tensor(3.14)

        res = Result(x)
        self.assertEqual(res.get_loss(), x)

        res.loss = torch.tensor(25.)
        self.assertEqual(res.get_loss(), x)

        if torch.cuda.is_available():
            x = torch.tensor(3.14).cuda()
            res = Result(x)

            res.detach_loss()
            self.assertEqual(res.get_loss()[0].is_cuda, False)
            self.assertEqual(x.is_cuda, True)

            gpu_res = Result(x, to_cpu=False)
            gpu_res.detach_loss()
            self.assertEqual(gpu_res.get_loss()[0].is_cuda, True)

    def test_set(self):
        x = torch.tensor(3.14)
        res = Result(x)

        res.detach_loss()
        self.assertEqual(torch.equal(res.get_loss()[0], torch.tensor(3.14)), True)

        res.abc = torch.ones(5,5,5)
        self.assertEqual(torch.equal(res.abc[0], torch.ones(5,5,5)), True)

        res.abc = (x, x)
        self.assertEqual(iterable_torch_eq(res.abc, [(x, x)]), True)

        res.abc = (torch.ones(5).cuda(), torch.ones(5).cuda())
        self.assertEqual(res.abc[0][1].is_cuda, False)

        if torch.cuda.is_available():
            gpu_res = Result(torch.tensor(5.), to_cpu=False)

            gpu_res.abc = (torch.ones(5).cuda(), torch.ones(5).cuda())
            self.assertEqual(gpu_res.abc[0][1].is_cuda, True)

    def test_collect(self):
        res1 = Result(torch.tensor(3.14))
        res1.loss = torch.tensor(97.)
        res1.a = torch.tensor(5.)
        res1.b = [torch.zeros(3,3)]
        res1.c = [torch.tensor(5.), torch.tensor(4.)]
        res1.detach_loss()

        res2 = Result((torch.tensor(1.), torch.tensor(2.)))
        res2.a = torch.tensor(5.)
        res2.b = torch.zeros(3,3)
        res2.c = [torch.tensor(5.), torch.tensor(4.)]
        res2.detach_loss()

        res3 = Result(torch.tensor(99.))
        res3.f = torch.zeros(5)
        res3.ab = [torch.ones(4)]
        res3.detach_loss()

        coll_res = Result.collect([res1, res2, res3])

        self.assertEqual(iterable_torch_eq(coll_res.a,
                                [torch.tensor(5.), torch.tensor(5.)]), True)
        self.assertEqual(iterable_torch_eq(coll_res.b,
                                [[torch.zeros(3,3)], torch.zeros(3,3)]), True)
        self.assertEqual(iterable_torch_eq(coll_res.c,
                                [[torch.tensor(5.), torch.tensor(4.)],
                                 [torch.tensor(5.), torch.tensor(4.)]]), True)
        self.assertEqual(iterable_torch_eq(coll_res.f, [torch.zeros(5)]), True)
        self.assertEqual(iterable_torch_eq(coll_res.ab, [[torch.ones(4)]]), True)
        self.assertEqual(iterable_torch_eq(coll_res.loss, [torch.tensor(3.14),
                                (torch.tensor(1.), torch.tensor(2.)),
                                 torch.tensor(99.)]), True)