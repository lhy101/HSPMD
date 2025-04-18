import hspmd
import hspmd.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys

# Warning: Remember to set rtol = 1e-05, atol = 3e-05 in `test_utils.py`

GRAD_TEST = True

class TestCeilOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_ceil_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestCeilOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.ceil(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.ceil(x)
            x.ceil_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.ceil(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.ceil(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.ceil(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.ceil_()
                hspmd_out = hspmd.ceil_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestFloorOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_floor_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestFloorOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.floor(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.floor(x)
            x.floor_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.floor(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.floor(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.floor(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.floor_()
                hspmd_out = hspmd.floor_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestRoundOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_round_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestRoundOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.round(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.round(x)
            x.round_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.round(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.round(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.round(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.round_()
                hspmd_out = hspmd.round_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestNegOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_neg_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNegOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.negative(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.neg(x)
            x.neg_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.neg(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.neg(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.neg(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.neg_()
                hspmd_out = hspmd.neg_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestPowOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    _exponent = [
        0.0,
        -1.0,
        -2.0, 
        4.0
    ]

    def test_pow_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPowOps._test_shapes:
            for exponent in TestPowOps._exponent:
                x_np = np.random.randn(*shape).astype(np.float64)
                gt = np.power(x_np, exponent)
                x = hspmd.from_numpy(x_np)
                res_out = hspmd.pow(x, exponent)
                x.pow_(exponent)
                self.assertTrue(allclose(res_out, gt))
                self.assertTrue(allclose(x, gt))
        # in-place pow backward is not supported by Hetu
        print(sys._getframe().f_code.co_name)

class TestReciprocalOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_reciprocal_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestReciprocalOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.reciprocal(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.reciprocal(x)
            x.reciprocal_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.reciprocal(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.reciprocal(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.reciprocal(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.reciprocal_()
                hspmd_out = hspmd.reciprocal_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestReluOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestReluOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.relu(x)
            x.relu_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.relu(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.relu(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.relu(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.relu_()
                hspmd_out = hspmd.relu_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestTanhOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_tanh_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTanhOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.tanh(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.tanh(x)
            x.tanh_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.tanh(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.tanh(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.tanh(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.tanh_()
                hspmd_out = hspmd.tanh_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestLeakyReluOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_leaky_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestLeakyReluOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hspmd.from_numpy(x_np)
                res_out = hspmd.leakyrelu(x, alpha)
                x.leakyrelu_(alpha)
                self.assertTrue(allclose(res_out, gt))
                self.assertTrue(allclose(x, gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                    hspmd_out = hspmd.leakyrelu(hspmd_in, alpha)
                    hspmd_loss = hspmd_out.sum()
                    hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hspmd_optimizer.minimize(hspmd_loss)
                    self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                    # in-place
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                    hspmd_out = hspmd_in.add(0)
                    # hspmd_out.leakyrelu_(alpha)
                    hspmd_out = hspmd.leakyrelu_(hspmd_out, alpha)
                    hspmd_loss = hspmd_out.sum()
                    hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hspmd_optimizer.minimize(hspmd_loss)
                    self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                    self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestSigmoidOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_sigmoid_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestSigmoidOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = 1 / (1 + np.exp(-x_np))
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.sigmoid(x)
            x.sigmoid_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sigmoid(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.sigmoid(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sigmoid(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.sigmoid_()
                hspmd_out = hspmd.sigmoid_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestSqrtOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_sqrt_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestSqrtOps._test_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            gt = np.sqrt(x_np)
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.sqrt(x)
            x.sqrt_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.sqrt(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.sqrt_()
                hspmd_out = hspmd.sqrt_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestRSqrtOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_rsqrt_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestRSqrtOps._test_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            gt = np.reciprocal(np.sqrt(x_np))
            x = hspmd.from_numpy(x_np)
            res_out = hspmd.rsqrt(x)
            x.rsqrt_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.rsqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd.rsqrt(hspmd_in)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.rsqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hspmd_in = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_out = hspmd_in.add(0)
                # hspmd_out.rsqrt_()
                hspmd_out = hspmd.rsqrt_(hspmd_out)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestWhereOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_where_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestWhereOps._test_shapes:
            cond_np = np.random.choice([0, 1], size=shape).astype(np.int64)
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            gt = np.where(cond_np, x_np, y_np)
            cond = hspmd.from_numpy(cond_np)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            res_out = hspmd.where(cond, x, y)
            hspmd.where_(cond, x, y)
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_cond = torch.tensor(cond_np, dtype=torch.bool)
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np)
                torch_out = torch.where(torch_cond, torch_x, torch_y)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hspmd_cond = hspmd.Tensor(cond_np)
                hspmd_x = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_y = hspmd.Tensor(y_np)
                hspmd_out = hspmd.where(hspmd_cond, hspmd_x, hspmd_y)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_x, torch_x.detach().numpy()))

                # in-place
                torch_cond = torch.tensor(cond_np, dtype=torch.bool)
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np)
                torch_out = torch.where(torch_cond, torch_x, torch_y)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hspmd_cond = hspmd.Tensor(cond_np)
                hspmd_x = hspmd.Tensor(x_np, requires_grad=True)
                hspmd_y = hspmd.Tensor(y_np)
                hspmd_out = hspmd_x.add(0)
                # hspmd_out.where_(hspmd_cond, hspmd_y)
                hspmd_out = hspmd.where_(hspmd_cond, hspmd_out, hspmd_y)
                hspmd_loss = hspmd_out.sum()
                hspmd_optimizer = hspmd.SGDOptimizer([hspmd_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hspmd_optimizer.minimize(hspmd_loss)
                self.assertTrue(allclose(hspmd_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hspmd_x, torch_x.detach().numpy()))
        print(sys._getframe().f_code.co_name)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hspmd.graph("eager"):
        with hspmd.context(eager_device="cuda:0"):
            unittest.main()
