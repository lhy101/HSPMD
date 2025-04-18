import hspmd
import hspmd.nn as nn
import numpy as np
import torch
import unittest
from test_utils import allclose

class TestArithmeticOps(unittest.TestCase):

    _test_elementwise_shapes = [
        (1024,), 
        (64, 256), 
        (64, 32, 16), 
    ]

    _test_broadcast_shapes = [
        ((1024,), (1,)), 
        ((1024,), (1024,)), 
        ((64, 256), (64, 1)), 
        ((64, 256), (1, 256)), 
        ((64, 256), (256,)), 
    ]

    def test_elementwise_add(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            c = np.random.randn()
            # tensor + tensor
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(hspmd.add(x, y), gt))
            # tensor + constant & constant + tensor
            gt = x_np + c
            self.assertTrue(allclose(x + c, gt))
            self.assertTrue(allclose(c + x, gt))
            self.assertTrue(allclose(x.add(c), gt))
            self.assertTrue(allclose(hspmd.add(x, c), gt))
            self.assertTrue(allclose(hspmd.add(c, x), gt))
    
    def test_broadcast_add(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(y + x, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(y.add(x), gt))
            self.assertTrue(allclose(hspmd.add(x, y), gt))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.add(torch_in, torch.from_numpy(x_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.add(hspmd_in, x)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_elementwise_sub(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            c = np.random.randn()
            # tensor - tensor
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hspmd.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hspmd.sub(y, x), gt))
            # tensor - constant
            gt = x_np - c
            self.assertTrue(allclose(x - c, gt))
            self.assertTrue(allclose(x.sub(c), gt))
            self.assertTrue(allclose(hspmd.sub(x, c), gt))
            # constant - tensor
            gt = c - x_np
            self.assertTrue(allclose(c - x, gt))
            self.assertTrue(allclose(hspmd.sub(c, x), gt))
    
    def test_broadcast_sub(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hspmd.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hspmd.sub(y, x), gt))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.sub(torch_in, torch.from_numpy(x_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.sub(hspmd_in, x)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.sub(torch.from_numpy(x_np), torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.sub(x, hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_neg(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            gt = np.negative(x_np)
            self.assertTrue(allclose(x.neg(), gt))
            self.assertTrue(allclose(hspmd.neg(x), gt))

    def test_elementwise_mul(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            c = np.random.randn()
            # tensor * tensor
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hspmd.mul(x, y), gt))
            # tensor * constant & constant * tensor
            gt = x_np * c
            self.assertTrue(allclose(x * c, gt))
            self.assertTrue(allclose(c * x, gt))
            self.assertTrue(allclose(x.mul(c), gt))
            self.assertTrue(allclose(hspmd.mul(x, c), gt))
            self.assertTrue(allclose(hspmd.mul(c, x), gt))
    
    def test_broadcast_mul(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hspmd.mul(x, y), gt))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.mul(torch_in, torch.from_numpy(x_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.mul(hspmd_in, x)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_elementwise_div(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            c = np.random.randn()
            # tensor / tensor
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hspmd.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hspmd.div(y, x), gt))
            # tensor - constant
            gt = x_np / c
            self.assertTrue(allclose(x / c, gt))
            self.assertTrue(allclose(x.div(c), gt))
            self.assertTrue(allclose(hspmd.div(x, c), gt))
            # constant - tensor
            gt = c / x_np
            self.assertTrue(allclose(c / x, gt))
            self.assertTrue(allclose(hspmd.div(c, x), gt))
    
    def test_broadcast_div(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hspmd.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hspmd.div(y, x), gt))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.div(torch_in, torch.from_numpy(x_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.div(hspmd_in, x)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

            torch_in = torch.tensor(y_np, requires_grad=True)
            torch_out = torch.div(torch.from_numpy(x_np), torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(y_np, trainable=True)
            hspmd_out = hspmd.div(x, hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_reciprocal(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            gt = np.reciprocal(x_np)
            self.assertTrue(allclose(x.reciprocal(), gt))
            self.assertTrue(allclose(hspmd.reciprocal(x), gt))

    def test_sqrt(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape))
            x = hspmd.from_numpy(x_np)
            gt = np.sqrt(x_np)
            self.assertTrue(allclose(x.sqrt(), gt))
            self.assertTrue(allclose(hspmd.sqrt(x), gt))

    def test_sum(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            z_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            z = hspmd.from_numpy(z_np)
            gt = x_np + y_np + z_np
            self.assertTrue(allclose(hspmd.sum([x,y,z]), gt))

class TestMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((64, 256), (256, 128)),
        ((8, 64, 256), (256, 128))
    ]
    
    def test_matmul_op(self):
        for shape_x, shape_y in TestMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            gt = np.matmul(x_np, y_np)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            self.assertTrue(allclose(hspmd.matmul2(x, y), gt))
            self.assertTrue(allclose(x.matmul2(y), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.matmul(torch_in, torch.from_numpy(y_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.matmul2(hspmd_in, y)
            hspmd_out.sum().backward()
            print(hspmd_in.grad.shape, torch_in.grad.numpy().shape)
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    # def test_linear_op(self):
    #     for shape_x, shape_y in TestMatMulOps._test_shapes:
    #         x_np = np.random.randn(*shape_x)
    #         w_np = np.random.randn(*shape_y)
    #         bias_np = np.random.randn(shape_y[-1])
    #         gt = np.matmul(x_np, w_np) + bias_np
    #         x = hspmd.from_numpy(x_np)
    #         w = hspmd.from_numpy(w_np)
    #         bias = hspmd.from_numpy(bias_np)
    #         torch_test = torch.addmm(torch.from_numpy(bias_np), torch.from_numpy(x_np), 
    #                     torch.from_numpy(w_np)).numpy()
    #         self.assertTrue(allclose(hspmd.linear(x, w, bias), gt))
    #         self.assertTrue(allclose(torch_test, gt))

    #         torch_in = torch.tensor(x_np, requires_grad=True)
    #         torch_out = torch.matmul(torch_in, torch.from_numpy(w_np)) + torch.from_numpy(bias_np)
    #         torch_out.sum().backward()
    #         hspmd_in = hspmd.Tensor(x_np, trainable=True)
    #         hspmd_out = hspmd.linear(hspmd_in, w, bias)
    #         hspmd_out.sum().backward()
    #         self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

class TestBatchMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((1, 64, 128), (1, 128, 32)),
        ((16, 64, 256), (16, 256, 128))
    ]
    
    def test_batch_matmul_op(self):
        for shape_x, shape_y in TestBatchMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            gt = torch.bmm(torch.from_numpy(x_np), torch.from_numpy(y_np)).numpy()
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            self.assertTrue(allclose(hspmd.bmm(x, y), gt))
            self.assertTrue(allclose(x.bmm(y), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.bmm(torch_in, torch.from_numpy(y_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.bmm(hspmd_in, y)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

# class TestMatDotOps(unittest.TestCase):

#     _test_shapes = [
#         ((128, 64), (128, 64)),
#         ((256, 64), (256, 16))
#     ]
    
#     def test_batch_matmul_op(self):
#         for shape_x, shape_y in TestMatDotOps._test_shapes:
#             x_np = np.random.randn(*shape_x)
#             y_np = np.random.randn(*shape_y)
#             x = hspmd.from_numpy(x_np)
#             y = hspmd.from_numpy(y_np)
#             gt = np.dot(x_np,y_np)
#             self.assertTrue(allclose(hspmd.dot(x, y), gt))
#             self.assertTrue(allclose(x.dot(y), gt))
    

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_sigmoid_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = 1 / (1 + np.exp(-x_np))
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.sigmoid(x), gt))
            self.assertTrue(allclose(x.sigmoid(), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.sigmoid(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.sigmoid(hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_sin_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.sin(x_np)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.sin(x), gt))
            self.assertTrue(allclose(x.sin(), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.sin(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.sin(hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_relu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape) - 0.5
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.relu(x), gt))
            self.assertTrue(allclose(x.relu(), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.relu(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.relu(hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
            
    
    def test_leaky_relu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hspmd.from_numpy(x_np)
                self.assertTrue(allclose(hspmd.leakyrelu(x, alpha), gt))
                self.assertTrue(allclose(x.leakyrelu(alpha), gt))

                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                torch_out.sum().backward()
                hspmd_in = hspmd.Tensor(x_np, trainable=True)
                hspmd_out = hspmd.leakyrelu(hspmd_in, alpha)
                hspmd_out.sum().backward()
                self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_tanh_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.tanh(x_np)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.tanh(x), gt))
            self.assertTrue(allclose(x.tanh(), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.tanh(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.tanh(hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_triu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.triu(torch.from_numpy(x_np), 0).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.triu(x, False, 0), gt))
            self.assertTrue(allclose(x.triu(False, 0), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.triu(torch_in, 0)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.triu(hspmd_in, False, 0)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_tril_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.tril(torch.from_numpy(x_np), 0).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.triu(x, True, 0), gt))
            self.assertTrue(allclose(x.triu(True, 0), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.tril(torch_in, 0)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.triu(hspmd_in, True, 0)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
            
    
    def test_softmax_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.softmax(torch.from_numpy(x_np), 1).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.softmax(x), gt))
            self.assertTrue(allclose(x.softmax(), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.softmax(torch_in, 1)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.softmax(hspmd_in)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))


class TestTransformOps(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (128, 128)
    ]

    _pad_shapes = [
        (8, 4, 32, 32),
        (16, 4, 16, 16)
    ]

    _transpose_shapes = [
        (16, 4, 16),
        (4, 8, 16, 32)
    ]

    def test_reshape_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            shape_to = list(shape)
            shape_to[0] = int(shape_to[0] / 2)
            shape_to[1] *= 2
            gt = np.reshape(x_np, tuple(shape_to))
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.reshape(x, shape_to), gt))
            self.assertTrue(allclose(x.reshape(shape_to), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.reshape(torch_in, tuple(shape_to)).contiguous()
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.reshape(hspmd_in, shape_to)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_broadcast_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            shape_to = list(shape)
            shape_to = [16] + shape_to
            gt = np.broadcast_to(x_np, tuple(shape_to))
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.broadcast(x, shape_to, []), gt))
            self.assertTrue(allclose(x.broadcast(shape_to, []), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.broadcast_to(torch_in, tuple(shape_to))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.broadcast(hspmd_in, shape_to, [])
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_concat_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            z_np = np.random.randn(*shape)
            gt = np.concatenate((x_np, y_np), 0)
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            z = hspmd.from_numpy(z_np)
            self.assertTrue(allclose(hspmd.concat(x, y, 0), gt))
            self.assertTrue(allclose(x.concat(y, 0), gt))
            self.assertTrue(allclose(hspmd.concat([x, y], 0), gt))
            gt = np.concatenate((x_np, y_np, z_np), 0)
            self.assertTrue(allclose(hspmd.concat([x, y, z], 0), gt))
    
    def test_pad_op(self):
        for shape in TestTransformOps._pad_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            gt = np.pad(x_np, ((0,0),(0,0),(1,1),(2,2)), "constant", constant_values = 0.1)
            self.assertTrue(allclose(hspmd.pad(x, [1,1,2,2], "constant", 0.1), gt))
            self.assertTrue(allclose(x.pad([1,1,2,2], "constant", 0.1), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.nn.functional.pad(torch_in, (0,0,0,0,1,1,2,2), "constant", 0.1)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.pad(hspmd_in, [1,1,2,2], "constant", 0.1)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_slice_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            begin_pos = list(np.random.randint(0, 16 ,size = [2]))
            out_size = list(np.random.randint(16, 32 ,size = [2]))
            gt = x_np[begin_pos[0]:begin_pos[0]+out_size[0], begin_pos[1]:begin_pos[1]+out_size[1]]
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.slice(x, begin_pos, out_size), gt))
            self.assertTrue(allclose(x.slice(begin_pos, out_size), gt))

    def test_split_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            idx = list(np.random.randint(0, 8 ,size = [1]))
            gt = np.split(x_np, 8, 0)[idx[0]]
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.split(x, [0], idx, [8]), gt))
            self.assertTrue(allclose(x.split([0], idx, [8]), gt))
    
    def test_transpose_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.transpose(x_np, (1, 0))
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.transpose(x, [1, 0]), gt))
            self.assertTrue(allclose(x.transpose([1, 0]), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.transpose(torch_in, 1, 0)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.transpose(hspmd_in, [1, 0])
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

        for shape in TestTransformOps._transpose_shapes:
            x_np = np.random.randn(*shape)
            perm = np.arange(x_np.ndim)
            np.random.shuffle(perm)
            perm = list(perm)
            gt = np.transpose(x_np, perm)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.transpose(x, perm), gt))
            self.assertTrue(allclose(x.transpose(perm), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch_in.permute(perm).contiguous()
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.transpose(hspmd_in, perm)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

class TestConv2dOps(unittest.TestCase):

    _data_shapes = [
        (4, 3, 16, 16),        
    ]

    _filter_shapes = [
        (3, 3, 2, 2),
        (4, 3, 4, 4)        
    ]

    def test_conv2d_op(self):
        for shape in TestConv2dOps._data_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape)
                f = hspmd.from_numpy(f_np)
                bias_np = np.random.randn()
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), stride = 1, padding = 0).numpy()
                bias_shape = [f_shape[0]]
                self.assertTrue(allclose(hspmd.conv2d(x, f, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, 0, 1), gt))
                # test conv2d add bias
                bias_np = np.random.randn(*bias_shape)
                bias = hspmd.from_numpy(bias_np)
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(bias_np), stride = 1, padding = 0).numpy()
                self.assertTrue(allclose(hspmd.conv2d(x, f, bias, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, bias, 0, 1), gt))

                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.conv2d(torch_in, torch.from_numpy(f_np), stride = 1, padding = 0)
                torch_out.sum().backward()
                hspmd_in = hspmd.Tensor(x_np, trainable=True)
                hspmd_out = hspmd.conv2d(hspmd_in, f, bias, 0, 1)
                hspmd_out.sum().backward()
                self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))



class TestPoolOps(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_maxpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            maxpool2d = torch.nn.MaxPool2d(2, 1, 0)
            gt = maxpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hspmd.maxpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.maxpool(2, 2, 0, 1), gt))
            
            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = maxpool2d(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.maxpool(hspmd_in, 2, 2, 0, 1)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_avgpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hspmd.from_numpy(x_np)
            avgpool2d = torch.nn.AvgPool2d(2, 1, 0)
            gt = avgpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hspmd.avgpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.avgpool(2, 2, 0, 1), gt))
            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = avgpool2d(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.avgpool(hspmd_in, 2, 2, 0, 1)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

class TestNormOps(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_batchnorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hspmd.from_numpy(x_np)
            scale_np = np.ones(shape[1]).astype(np.float32)
            scale = hspmd.from_numpy(scale_np)
            bias_np = np.zeros(shape[1]).astype(np.float32)
            bias = hspmd.from_numpy(bias_np)
            running_mean_np = np.empty(shape[1]).astype(np.float32)
            running_mean = hspmd.from_numpy(running_mean_np)
            running_var_np = np.empty(shape[1]).astype(np.float32)
            running_var = hspmd.from_numpy(running_var_np)
            save_mean_np = np.empty(shape[1]).astype(np.float32)
            save_mean = hspmd.from_numpy(save_mean_np)
            save_var_np = np.empty(shape[1]).astype(np.float32)
            save_var = hspmd.from_numpy(save_var_np)
            gt = torch.batch_norm(torch.from_numpy(x_np), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                 running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True).numpy()
            self.assertTrue(allclose(hspmd.batch_norm(x, scale, bias, running_mean, running_var, 0.1 ,1e-5)[0], gt))
            self.assertTrue(allclose(x.batch_norm(scale, bias, running_mean, running_var, 0.1 ,1e-5)[0], gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.batch_norm(torch_in, weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                        running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.batch_norm(hspmd_in, scale, bias, running_mean, running_var, 0.1 ,1e-5)[0]
            hspmd_out.sum().backward()
            # self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_layernorm_op(self):
        for shape in TestPoolOps._test_shapes:
            norm_shape = shape[3:]
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hspmd.from_numpy(x_np)
            scale_np = np.ones(norm_shape).astype(np.float32)
            scale = hspmd.from_numpy(scale_np)
            bias_np = np.zeros(norm_shape).astype(np.float32)
            bias = hspmd.from_numpy(bias_np)
            layernorm = torch.nn.LayerNorm(norm_shape, 1e-5)
            gt = layernorm(torch.from_numpy(x_np)).detach().numpy()
            gt2 = torch.layer_norm(torch.from_numpy(x_np), normalized_shape=tuple(norm_shape), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                  eps=1e-5).numpy()
            self.assertTrue(allclose(gt2, gt))
            self.assertTrue(allclose(hspmd.layer_norm(x, scale, bias, 1e-5)[0], gt))
            self.assertTrue(allclose(x.layer_norm(scale, bias, 1e-5)[0], gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = layernorm(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.layer_norm(hspmd_in, scale, bias, 1e-5)[0]
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_instancenorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hspmd.from_numpy(x_np)
            temp_shape = list(shape)
            # temp_shape[-1] = 1
            # temp_shape[-2] = 1
            temp_shape = [temp_shape[1]]
            temp_shape = tuple(temp_shape)
            save_mean_np = np.empty(temp_shape).astype(np.float32)
            save_mean = hspmd.from_numpy(save_mean_np)
            save_var_np = np.empty(temp_shape).astype(np.float32)
            save_var = hspmd.from_numpy(save_var_np)
            instancenorm = torch.nn.InstanceNorm2d(num_features=shape[1], eps=1e-5)
            gt = instancenorm(torch.from_numpy(x_np)).detach().numpy()
            self.assertTrue(allclose(hspmd.instance_norm(x, 1e-5)[0], gt))
            self.assertTrue(allclose(x.instance_norm(1e-5)[0], gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = instancenorm(torch_in)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.instance_norm(hspmd_in, 1e-5)[0]
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

class TestReduceOps(unittest.TestCase):

    _test_shapes = [
        (16, 4, 16, 16),
        (1, 8, 32, 32),
        (1,),
    ]
    
    def test_reduce_sum_op(self):
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = np.sum(x_np)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.reduce(x, "sum"), gt))
            self.assertTrue(allclose(x.reduce("sum"), gt))
            self.assertTrue(allclose(hspmd.sum(x), gt))
            self.assertTrue(allclose(x.sum(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.sum(x_np, tuple(axes))
                x = hspmd.from_numpy(x_np)
                self.assertTrue(allclose(hspmd.sum(x, axes), gt))
                self.assertTrue(allclose(x.sum(axes), gt))
                #keepdim test
                gt = np.sum(x_np, tuple(axes), keepdims=True)
                x = hspmd.from_numpy(x_np)
                self.assertTrue(allclose(hspmd.sum(x, axes, [True]), gt))
                

    def test_reduce_mean_op(self):
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = np.average(x_np)
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.reduce(x, "mean"), gt))
            self.assertTrue(allclose(x.reduce("mean"), gt))
            self.assertTrue(allclose(hspmd.mean(x), gt))
            self.assertTrue(allclose(x.mean(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.average(x_np, tuple(axes))
                x = hspmd.from_numpy(x_np)
                self.assertTrue(allclose(hspmd.mean(x, axes), gt))
                self.assertTrue(allclose(x.mean(axes), gt))
                #keepdim test
                gt = np.mean(x_np, tuple(axes), keepdims=True)
                x = hspmd.from_numpy(x_np)
                self.assertTrue(allclose(hspmd.mean(x, axes, [True]), gt))

class TestLossOps(unittest.TestCase):
    _test_binary_label_shapes = [
        (64, 1)
    ]

    _test_nllloss_label_shapes = [
        ((64, 16), (64, ))
    ]

    _test_cross_entropy_label_shapes = [
        (64, 16)
    ]

    def test_bce_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            bce = torch.nn.BCELoss(reduction="mean")
            # gt = torch.nn.functional.binary_cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            gt =bce(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_np)
            loss = hspmd.binary_cross_entropy(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = bce(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.binary_cross_entropy(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_nllloss_op(self):
        for shape, lshape in TestLossOps._test_nllloss_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=lshape).astype(np.int64)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            #gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_np)
            loss = hspmd.nll_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = torch.nn.functional.nll_loss(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.nll_loss(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_kldivloss_op(self):
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.kl_div(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_np)
            loss = hspmd.kl_div(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = torch.nn.functional.kl_div(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.kl_div(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_mseloss_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.mse_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_np)
            loss = hspmd.mse_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = torch.nn.functional.mse_loss(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.mse_loss(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_softmax_cross_entropy_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_cross_entropy_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            #labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=(64,)).astype(np.int64)
            labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels_np), 16).numpy().astype(np.float32)
            # probs_np = np.arange(4).astype(np.float32) + 1
            # probs_np = probs_np.reshape(2,2)
            # labels_np = np.array([[1,0],[0,1]]).astype(np.float32).reshape(2,2)
            # crs_etp = torch.nn.CrossEntropyLoss()
            # gt = crs_etp(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            gt = torch.nn.functional.cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_onehot)
            loss = hspmd.softmax_cross_entropy(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = torch.nn.functional.cross_entropy(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.softmax_cross_entropy(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_softmax_cross_entropy_sparse_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_cross_entropy_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            #labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=(64,)).astype(np.int64)
            # labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels_np), 16).numpy().astype(np.float32)
            gt = torch.nn.functional.cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hspmd.from_numpy(probs_np)
            labels = hspmd.from_numpy(labels_np)
            loss = hspmd.softmax_cross_entropy_sparse(probs, labels)
            self.assertTrue(allclose(loss, gt))

            torch_in = torch.tensor(probs_np, requires_grad=True)
            torch_out = torch.nn.functional.cross_entropy(torch_in, torch.from_numpy(labels_np))
            torch_out.backward()
            hspmd_in = hspmd.Tensor(probs_np, trainable=True)
            hspmd_out = hspmd.softmax_cross_entropy_sparse(hspmd_in, labels)
            hspmd_out.backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

class TestEinsumOps(unittest.TestCase):

    _test_args = [
        ("ij->ji",((64, 32),)),
        ("ij,ij->ij", ((64, 32), (64, 32))),
        ("ii->i",((64, 64),)),
        ("...ij->...ji",((64, 32, 4, 2, 4),)),
        ("ij->",((64, 32),)),
        ("ij->j",((64, 32),)),
        ("ik,k",((64, 32),(32,))),
        ("ik,kj",((64, 32),(32, 16))),
        ("i,i",((2,),(2,))),
        ("ij,ij",((64, 32),(64, 32))),
        ("i,j",((64, ),(32, ))),
        ("ijk,ikl->ijl",((64, 32, 16), (64, 16, 24))),
        ("pqrs,tuqvr->pstuv", ((4, 5, 6, 8), (9, 7, 5, 13, 6))),
        ("ik,jkl,il->ij",((64, 32), (16, 32, 48), (64, 48))),
        ("ijk",((64, 32, 16),)),
        ("b n h w, n d -> b d h w",((64, 32, 8, 4), (32, 16))),
        ("n d, n d -> n",((64, 32), (64, 32))),
        ("i d, j d -> i j",((64, 32), (48, 32))),
        ("b h i d, b h j d -> b h i j",((64, 32, 4, 8), (64, 32, 6, 8))),
        ("b h i j, b h j d -> b h i d",((64, 32, 4, 8), (64, 32, 8, 6))),
        ("b i d, b i j d -> b i j",((64, 32, 4), (64, 32, 8, 4))),
        ("b x i d, b j d -> b x i j",((64, 32, 4, 8), (64, 5, 8))),
        ("b x i j, b j d -> b x i d",((64, 32, 4, 5), (64, 5, 8))),
        ("hij, ijc->ihc",((64, 32, 16), (32, 16, 8))),
        ("rac,rab->rbc",((64, 32, 4), (64, 32, 7))),
        ("ra,rab->rb",((64, 32), (64, 32, 8))),
        ("qhc,khc->qkh",((64, 32, 4), (48, 32, 4))),
        ("nm, mrc->nrc",((64, 32), (32, 8, 6))),
        ("abc,adc->bdc",((64, 32, 15), (64, 13, 15))),
        ("dceb,cef->dbf",((64, 32, 4, 8), (32, 4, 13))),
        ("acb,ade->dceb",((64, 32, 7), (64, 15, 9))),
        ("qkc,ch->hqk",((64, 32, 4), (4, 13))),
        ("bhqk,bkhc->bqhc",((64, 32, 4, 8), (64, 8, 32, 7))),
        ("bqa,ahc->bqhc",((64, 32, 8), (8, 15, 9))),
        ("...lc, ...c -> ...l",((64, 32, 7), (64, 7))),
        ("...lc, ...lc -> ...l",((64, 32, 7), (64, 32, 7))),
        ("...id,...jd->...ij",((64, 32, 4, 8), (64, 32, 5, 8))),
        ("...klm,kmn->...kln",((64, 32, 4, 8), (32, 8, 11))),
        ("...ikl, ...jk -> ...ijl",((64, 32, 4, 8), (64, 15, 4))),
        ("...l,...l->...",((64, 32, 17), (64, 32, 17))),
        ("ijk,ijk...->ij...",((64, 32, 4), (64, 32, 4, 9))),
        ("bxi,oij,byj->boxy",((64, 32, 5), (17, 5, 13), (64, 9, 13))),
        ("ijac,ijkp->ijakcp",((64, 32, 4, 8), (64, 32, 5, 7))),
        ("cdij,cbi->cdbj",((64, 32, 4, 8), (64, 19, 4))),
        ("bsid,bsjd->bijd",((64, 32, 4, 8), (64, 32, 17, 8))),
        ("bsid,bsje->bijde",((64, 32, 4, 8), (64, 32, 17, 9))),
        ("...bac,...dae->...bdce",((64, 32, 4, 8), (64, 19, 4, 5))),
        ("...abc,...adc->...bdc",((64, 32, 4, 8), (64, 32, 7, 8))),
        ("...qhd,...khd->...hqk",((64, 32, 4, 8), (64, 23, 4, 8))),
        ("...vhf,...qhv->...qhf",((64, 32, 4, 8), (64, 19, 4, 32))),
        ("...ij,jk->ik",((64, 32, 4, 8), (8, 13))),
    ]
    
    def test_einsum_op_simple(self):
        for equation, nshapes in TestEinsumOps._test_args:
            inputs_np = []
            inputs_hspmd = []
            for shape in nshapes:
                input_np = np.random.randn(*shape) * 10
                input_hspmd = hspmd.from_numpy(input_np)
                inputs_np.append(torch.from_numpy(input_np))
                inputs_hspmd.append(input_hspmd)
            gt = torch.einsum(equation, *inputs_np).numpy()
            self.assertTrue(allclose(hspmd.einsum(equation, inputs_hspmd), gt))

class TestOtherOps(unittest.TestCase):

    _asstrided_test_shapes = [
        ((8, 8), (4, 4), (1, 2)),
        ((6, 4, 6, 8), (2, 3, 4, 5), (1, 2, 1, 1))
    ]

    _embedding_test_shapes = [
        ((4, 4), (5)),
        ((16, 32), (16))
    ]

    _interpolate_test_shapes = [
        ((1, 1, 2, 2), (4, 4)),
        ((3, 4, 5, 5), (20, 20))
    ]

    _maskedfill_test_shapes = [
        ((3, 4, 5, 6),),
        ((1, 9, 1, 10),)
    ]

    _norm_test_shapes = [
        ((4, 5, 2, 3), 2, 2),
        ((3, 4, 5, 5), 0, 1)
    ]

    _repeat_test_shapes = [
        ((3, 5, 7), (2, 2, 3, 4)),
        ((2, 4, 6, 8), (2, 3, 4, 5) )
    ]

    _roll_test_shapes = [
        ((2, 2), (1,), (0,)),
        ((3, 6, 7, 9), (2, 4, 6), (0, 1, 3)),
        ((2, 4, 6, 8), (1, 7), (2, 3) )
    ]

    _gather_test_shapes = [
        ((2, 2), (2, 1), 1),
        ((5, 16, 32), (1, 16, 32), 0)
    ]

    _onehot_test_shapes = [
        (32, 4),
        (64,)
    ]

    def test_arangeop(self):
        gt = torch.arange(0, 100, 4).numpy()
        self.assertTrue(allclose(hspmd.arange(0, 100, 4), gt))
    
    def test_asstridedop(self):
        for shape_x, shape_y, stride in TestOtherOps._asstrided_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.as_strided(torch.from_numpy(x_np), shape_y, stride).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.as_strided(x, list(shape_y), list(stride)), gt))
            self.assertTrue(allclose(x.as_strided(list(shape_y), list(stride)), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.as_strided(torch_in, shape_y, stride)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.as_strided(hspmd_in, list(shape_y), list(stride))
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_gatherop(self):
        for shape_x, shape_id, dim in TestOtherOps._gather_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[dim], size=shape_id)
            gt = torch.gather(torch.from_numpy(x_np), dim, torch.from_numpy(id_np)).numpy()
            x = hspmd.from_numpy(x_np)
            id = hspmd.from_numpy(id_np)
            self.assertTrue(allclose(hspmd.gather(x, dim, id), gt))
            self.assertTrue(allclose(x.gather(dim, id), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.gather(torch_in, dim, torch.from_numpy(id_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.gather(hspmd_in, dim, id)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_interpolateop(self):
        for shape_x, shape_o in TestOtherOps._interpolate_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.nn.functional.interpolate(torch.from_numpy(x_np), shape_o, mode='bicubic').numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.interpolate(x, list(shape_o)), gt))
            self.assertTrue(allclose(x.interpolate(list(shape_o)), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.nn.functional.interpolate(torch_in, shape_o, mode='bicubic')
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.interpolate(hspmd_in, list(shape_o))
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_maskedfillop(self):
        for shape_x in TestOtherOps._maskedfill_test_shapes:
            shape_x = shape_x[0]
            x_np = np.random.randn(*shape_x)
            mask_np = np.random.choice([0, 1], size=shape_x).astype(np.int64)
            val = np.random.random()
            gt = torch.masked_fill(torch.from_numpy(x_np), torch.from_numpy(mask_np), val).numpy()
            x = hspmd.from_numpy(x_np)
            mask = hspmd.from_numpy(mask_np)
            self.assertTrue(allclose(hspmd.masked_fill(x, mask, val), gt))
            self.assertTrue(allclose(x.masked_fill(mask, val), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.masked_fill(torch_in, torch.from_numpy(mask_np), val)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.masked_fill(hspmd_in, mask)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_normop(self):
        for shape_x, dim0, p0 in TestOtherOps._norm_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.norm(torch.from_numpy(x_np), p=p0, dim=dim0).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.norm(x, p0, dim0), gt))
            self.assertTrue(allclose(x.norm(p0, dim0), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.norm(torch_in, p=p0, dim=dim0)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.norm(hspmd_in, p0, dim0)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_repeatop(self):
        for shape_x, repeats in TestOtherOps._repeat_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.from_numpy(x_np).repeat(*repeats).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.repeat(x, list(repeats)), gt))
            self.assertTrue(allclose(x.repeat(list(repeats)), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch_in.repeat(*repeats)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.repeat(hspmd_in, list(repeats))
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_rollop(self):
        for shape_x, shifts, dims in TestOtherOps._roll_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.roll(torch.from_numpy(x_np), shifts=shifts, dims=dims).numpy()
            x = hspmd.from_numpy(x_np)
            # print(hspmd.roll(x, list(shifts), list(dims)).numpy(force=True), "\n", gt)
            self.assertTrue(allclose(hspmd.roll(x, list(shifts), list(dims)), gt))
            self.assertTrue(allclose(x.roll(list(shifts), list(dims)), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.roll(torch_in, shifts, dims)
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.roll(hspmd_in, list(shifts), list(dims))
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))
    
    def test_embedding_lookupop(self):
        for shape_x, shape_id in TestOtherOps._embedding_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[0], size=shape_id)
            gt = torch.embedding(torch.from_numpy(x_np), torch.from_numpy(id_np)).numpy()
            x = hspmd.from_numpy(x_np)
            id = hspmd.from_numpy(id_np)
            self.assertTrue(allclose(hspmd.embedding_lookup(x, id), gt))
            self.assertTrue(allclose(x.embedding_lookup(id), gt))

            torch_in = torch.tensor(x_np, requires_grad=True)
            torch_out = torch.embedding(torch_in, torch.from_numpy(id_np))
            torch_out.sum().backward()
            hspmd_in = hspmd.Tensor(x_np, trainable=True)
            hspmd_out = hspmd.embedding_lookup(hspmd_in, id)
            hspmd_out.sum().backward()
            self.assertTrue(allclose(hspmd_in.grad, torch_in.grad.numpy()))

    def test_onehotop(self):
        for shape_x in TestOtherOps._onehot_test_shapes:
            x_np = np.random.randint(0, 16, size=shape_x)
            gt = torch.nn.functional.one_hot(torch.from_numpy(x_np), num_classes = 16).numpy()
            x = hspmd.from_numpy(x_np)
            self.assertTrue(allclose(hspmd.onehot(x, 16), gt))
            self.assertTrue(allclose(x.onehot(16), gt))

    def test_whereop(self):
        for shape_x in TestOtherOps._onehot_test_shapes:
            cond_np = np.random.choice([0, 1], size=shape_x).astype(np.int64)
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_x)
            gt = np.where(cond_np, x_np, y_np)
            cond = hspmd.from_numpy(cond_np) 
            x = hspmd.from_numpy(x_np)
            y = hspmd.from_numpy(y_np)
            self.assertTrue(allclose(hspmd.where(cond, x, y), gt))

                

if __name__ == "__main__":
    unittest.main()
