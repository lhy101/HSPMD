import hspmd
import numpy as np
import torch
import unittest

class HetuMLP(hspmd.nn.Module):
    
    def __init__(self, weights_np):
        super(HetuMLP, self).__init__()
        for i, w_np in enumerate(weights_np):
            # TODO: set trainable inside make subclass
            self.register_parameter(
                f"w{i}", 
                hspmd.nn.Parameter(hspmd.Tensor(w_np, trainable=True)))
        self.num_layers = len(weights_np)
    
    def forward(self, x):
        act = x
        for i in range(self.num_layers):
            w = self.__getattr__(f"w{i}")
            # act = hspmd.matmul(act, w)
            with hspmd.context(extra_deps=[w]):
                act = act.matmul(w)
            if i + 1 < self.num_layers:
                act = hspmd.relu(act)
        return act

class TorchMLP(torch.nn.Module):
    
    def __init__(self, weights_np):
        super(TorchMLP, self).__init__()
        for i, w_np in enumerate(weights_np):
            self.register_parameter(
                f"w{i}", 
                torch.nn.Parameter(torch.tensor(w_np)))
        self.num_layers = len(weights_np)
    
    def forward(self, x):
        act = x
        for i in range(self.num_layers):
            w = self.__getattr__(f"w{i}")
            act = torch.matmul(act, w)
            if i + 1 < self.num_layers:
                act = act.relu()
        return act

class HetuCNN(hspmd.nn.Module):
    
    def __init__(self, weights):
        super(HetuCNN, self).__init__()

        self.batchnorm = hspmd.nn.InstanceNorm(32, eps=0) #hspmd.nn.BatchNorm(32)

        self.dropout1 = hspmd.nn.Dropout(0.25)
        self.dropout2 = hspmd.nn.Dropout(0.5)
 
        self.register_parameter(
                f"filter1", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[0], trainable=True)))
        self.register_parameter(
                f"bias1", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[1], trainable=True)))
        self.register_parameter(
                f"filter2", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[2], trainable=True)))
        self.register_parameter(
                f"bias2", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[3], trainable=True)))
        self.register_parameter(
                f"matmul1", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[4], trainable=True)))
        self.register_parameter(
                f"bias3", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[5], trainable=True)))
        self.register_parameter(
                f"matmul2", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[6], trainable=True)))
        self.register_parameter(
                f"bias4", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[7], trainable=True)))
    
    def forward(self, x):

        out = self.batchnorm(x)

        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = hspmd.conv2d(out, f, b, 0, 1)
        out = hspmd.relu(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = hspmd.conv2d(out, f, b, 0, 1)
        out = hspmd.relu(out)
 
        out = hspmd.maxpool(out, 2, 2, 0, 2)
        # # out = self.dropout1(out)
         
        out = hspmd.reshape(out, [out.shape[0], 9216])
         
        f = self.__getattr__(f"matmul1")
        b = self.__getattr__(f"bias3")
        out = hspmd.linear(out, f, b)
        out = hspmd.relu(out)
 
        # # [b, 128] => [b, 10]
        # # out = self.dropout2(out)
        # # out = self.fc2(out)
        f = self.__getattr__(f"matmul2")
        b = self.__getattr__(f"bias4")
        out = hspmd.linear(out, f, b)
 
        # out = hspmd.softmax(out)
        out = hspmd.sigmoid(out)

        return out

class TorchCNN(torch.nn.Module):
    
    def __init__(self, weights):
        super(TorchCNN, self).__init__()

        # self.dropout1 = torch.nn.Dropout(0.25)
        # self.dropout2 = torch.nn.Dropout(0.5)
        self.batchnorm = torch.nn.InstanceNorm2d(32, eps=0) #torch.nn.BatchNorm2d(32)
 
        self.register_parameter(
                f"filter1", 
                torch.nn.Parameter(torch.tensor(weights[0])))
        self.register_parameter(
                f"bias1", 
                torch.nn.Parameter(torch.tensor(weights[1])))
        self.register_parameter(
                f"filter2", 
                torch.nn.Parameter(torch.tensor(weights[2])))
        self.register_parameter(
                f"bias2", 
                torch.nn.Parameter(torch.tensor(weights[3])))
        self.register_parameter(
                f"matmul1", 
                torch.nn.Parameter(torch.tensor(weights[4])))
        self.register_parameter(
                f"bias3", 
                torch.nn.Parameter(torch.tensor(weights[5])))
        self.register_parameter(
                f"matmul2", 
                torch.nn.Parameter(torch.tensor(weights[6])))
        self.register_parameter(
                f"bias4", 
                torch.nn.Parameter(torch.tensor(weights[7])))
    
    def forward(self, x):

        out = self.batchnorm(x)

        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = torch.conv2d(out, f, b, stride = 1, padding = 0)
        out = torch.relu(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = torch.conv2d(out, f, b, stride = 1, padding = 0)
        out = torch.relu(out)

        maxpool2d = torch.nn.MaxPool2d(2, 2, 0)
        out = maxpool2d(out)
        # # out = self.dropout1(out)
         
        out = torch.reshape(out, [out.shape[0], 9216])
         
        f = self.__getattr__(f"matmul1")
        b = self.__getattr__(f"bias3")
        out = torch.matmul(out, f) + b
        out = torch.relu(out)
 
        # # [b, 128] => [b, 10]
        # # out = self.dropout2(out)
        # # out = self.fc2(out)
        f = self.__getattr__(f"matmul2")
        b = self.__getattr__(f"bias4")
        out = torch.matmul(out, f) + b
 
        # out = torch.softmax(out, dim = 1)
        out = torch.sigmoid(out)
        return out

class Hetu_ResidualBlock(hspmd.nn.Module):
    def __init__(self, weights, outchannel):
        super(Hetu_ResidualBlock, self).__init__()

        self.register_parameter(
                f"filter1", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[0], trainable=True)))
        self.register_parameter(
                f"bias1", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[1], trainable=True)))
        self.register_parameter(
                f"filter2", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[2], trainable=True)))
        self.register_parameter(
                f"bias2", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[3], trainable=True)))
        self.register_parameter(
                f"filter3", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[4], trainable=True)))
        self.register_parameter(
                f"bias3", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[5], trainable=True)))
        self.register_parameter(
                f"weight", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[6], trainable=True)))
        self.register_parameter(
                f"bias4", 
                hspmd.nn.Parameter(hspmd.Tensor(weights[7], trainable=True)))
        
        self.norm1 = hspmd.nn.InstanceNorm(outchannel, eps=1e-2)

        self.norm2 = hspmd.nn.InstanceNorm(outchannel, eps=1e-2)

        self.norm3 = hspmd.nn.InstanceNorm(outchannel, eps=1e-2)
    
    def shortcut(self, x):
        f = self.__getattr__(f"filter3")
        b = self.__getattr__(f"bias3")
        out = hspmd.conv2d(x, f, b, 0, 1)
        # out = self.norm3(out)

        return out

    def forward(self, x):
        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = hspmd.conv2d(x, f, b, 0, 1)
        # out = self.norm1(out)
        out = hspmd.relu(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = hspmd.conv2d(out, f, b, 0, 1)
        # out = self.norm2(out)
        out += self.shortcut(x)
        out = hspmd.relu(out)
        out = hspmd.reshape(out, [out.shape[0], 1728])

        f = self.__getattr__(f"weight")
        b = self.__getattr__(f"bias4")
        out = hspmd.linear(out, f, b)

        return out

class ResidualBlock_low(torch.nn.Module):
    def __init__(self, weights, outchannel):
        super(ResidualBlock_low, self).__init__()
        self.register_parameter(
                f"filter1", 
                torch.nn.Parameter(torch.tensor(weights[0])))
        self.register_parameter(
                f"bias1", 
                torch.nn.Parameter(torch.tensor(weights[1])))
        self.register_parameter(
                f"filter2", 
                torch.nn.Parameter(torch.tensor(weights[2])))
        self.register_parameter(
                f"bias2", 
                torch.nn.Parameter(torch.tensor(weights[3])))
        self.register_parameter(
                f"filter3", 
                torch.nn.Parameter(torch.tensor(weights[4])))
        self.register_parameter(
                f"bias3", 
                torch.nn.Parameter(torch.tensor(weights[5])))
        self.register_parameter(
                f"weight", 
                torch.nn.Parameter(torch.tensor(weights[6])))
        self.register_parameter(
                f"bias4", 
                torch.nn.Parameter(torch.tensor(weights[7])))
        
        self.norm1 = torch.nn.InstanceNorm2d(outchannel, eps=1e-2)

        self.norm2 = torch.nn.InstanceNorm2d(outchannel, eps=1e-2)

        self.norm3 = torch.nn.InstanceNorm2d(outchannel, eps=1e-2)
 
    def shortcut(self, x):
        f = self.__getattr__(f"filter3")
        b = self.__getattr__(f"bias3")
        out = torch.conv2d(x, f, b, padding=0, stride=1)
        out = self.norm3(out)

        return out

    def forward(self, x):
        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = torch.conv2d(x, f, b, padding=0, stride=1)
        # out = self.norm1(out)
        out = torch.relu(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = torch.conv2d(out, f, b, padding=0, stride=1)
        # out = self.norm2(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        out = torch.reshape(out, (out.shape[0], 1728))

        f = self.__getattr__(f"weight")
        b = self.__getattr__(f"bias4")
        out = torch.matmul(out, f) + b
        return out



# class TestMLP(unittest.TestCase):

#     def test_dar_mode(self):
#         weights_np = []
#         dims = [64, 256, 8192 ,128, 10, 1]
#         # dims = [10, 1]
#         for i in range(1, len(dims)):
#             w = np.random.uniform(0.0, 1.0, size=(dims[i - 1], dims[i]))
#             w = ((w - 0.5) * (2.0 / np.sqrt(dims[i - 1]))).astype(np.float32)
#             # w = np.random.randn(dims[i - 1], dims[i]).astype(np.float32)
#             weights_np.append(w)

#         hspmd_model = HetuMLP(weights_np)
#         x = hspmd.placeholder(hspmd.float32, shape=[-1, dims[0]])
#         y = hspmd.placeholder(hspmd.float32, shape=[-1, 1])
#         pred = hspmd_model(x)
#         probs = hspmd.sigmoid(pred)
#         loss = hspmd.binary_cross_entropy(probs, y, "mean")

#         optimizer = hspmd.optim.SGD(lr=0.1)
#         train_op = optimizer.minimize(loss)

#         torch_model = TorchMLP(weights_np)
#         criterion_torch = torch.nn.BCELoss(reduction="mean")
#         optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)

#         executor = hspmd.DARExecutor("cuda:0")

#         for i in range(100):
#             x_val = np.random.randn(16, dims[0]).astype(np.float32)
#             y_val = np.random.choice([0, 1], size=(16, 1)).astype(np.float32)

#             loss_val, _ = executor.run([loss, train_op], feed_dict={
#                 x: x_val, 
#                 y: y_val, 
#             })

#             x_torch = torch.from_numpy(x_val)
#             y_torch = torch.from_numpy(y_val)
#             preds_torch = torch_model(x_torch)
#             probs_torch = preds_torch.sigmoid()
#             loss_torch = criterion_torch(probs_torch, y_torch)
#             optimizer_torch.zero_grad()
#             loss_torch.sum().backward()
#             optimizer_torch.step()


#             diffs, ratios = [], []
#             for w1, w2 in zip(hspmd_model.parameters(), torch_model.parameters()):
#                 w1 = w1.numpy(force=True)
#                 w2 = w2.detach().numpy()
#                 assert w1.shape == w2.shape
#                 diff = np.abs(w1 - w2).sum()
#                 ratio = (np.abs((w1 - w2) / w2)).sum()
#                 diffs.append(diff)
#                 ratios.append(ratio)
#             print(diffs, ratios)

    # def test_dbr_mode(self):
    #     # return
    #     weights_np = []
    #     dims = [256, 64, 16, 1]
    #     # dims = [10, 1]
    #     for i in range(1, len(dims)):
    #         w = np.random.uniform(0.0, 1.0, size=(dims[i - 1], dims[i]))
    #         w = ((w - 0.5) * (2.0 / np.sqrt(dims[i - 1]))).astype(np.float32)
    #         # w = np.random.randn(dims[i - 1], dims[i]).astype(np.float32)
    #         weights_np.append(w)

    #     hspmd_model = HetuMLP(weights_np)
    #     optimizer = hspmd.optim.SGD(hspmd_model.parameters(), lr=0.1)
        
    #     torch_model = TorchMLP(weights_np)
    #     criterion_torch = torch.nn.BCELoss(reduction='none')
    #     optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)
        
    #     for i in range(100):
    #         x_np = np.random.randn(16, dims[0]).astype(np.float32)
    #         y_np = np.random.choice([0, 1], size=(16, 1)).astype(np.float32)

    #         x = hspmd.from_numpy(x_np)
    #         y = hspmd.from_numpy(y_np)
            
    #         preds = hspmd_model(x)
    #         probs = hspmd.sigmoid(preds)
    #         loss = hspmd.binary_cross_entropy(probs, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         x_torch = torch.from_numpy(x_np)
    #         y_torch = torch.from_numpy(y_np)

    #         preds_torch = torch_model(x_torch)
    #         probs_torch = preds_torch.sigmoid()
    #         loss_torch = criterion_torch(probs_torch, y_torch).sum()
    #         optimizer_torch.zero_grad()
    #         loss_torch.backward()
    #         optimizer_torch.step()
            
    #         diffs, ratios = [], []
    #         for w1, w2 in zip(hspmd_model.parameters(), torch_model.parameters()):
    #             w1 = w1.numpy(force=True)
    #             w2 = w2.detach().numpy()
    #             assert w1.shape == w2.shape
    #             diff = np.abs(w1 - w2).sum()
    #             ratio = (np.abs((w1 - w2) / w2)).sum()
    #             diffs.append(diff)
    #             ratios.append(ratio)
    #         print(diffs, ratios)





# class TestCNN(unittest.TestCase):

#     def test_dar_mode(self):
#         weights_np = [np.random.uniform(-0.005, 0.005, size=(32, 1, 3, 3)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(32,)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(64, 32, 3, 3)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(64,)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(9216, 128)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(128,)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(128, 10)).astype(np.float32),
#                       np.random.uniform(-0.005, 0.005, size=(10,)).astype(np.float32),]

#         hspmd_model = HetuCNN(weights_np)
#         x = hspmd.placeholder(hspmd.float32, shape=[-1, 1, 28, 28])
#         y = hspmd.placeholder(hspmd.float32, shape=[-1, 10])
#         pred = hspmd_model(x)
#         loss = hspmd.softmax_cross_entropy(pred, y)
#         # probs = hspmd.sigmoid(pred)
#         # loss = hspmd.binary_cross_entropy(probs, y, "mean")

#         optimizer = hspmd.optim.SGD(lr=0.1)
#         train_op = optimizer.minimize(loss)

#         torch_model = TorchCNN(weights_np)
#         criterion_torch = torch.nn.CrossEntropyLoss()
#         optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)

#         executor = hspmd.DARExecutor("cuda:0")

#         for i in range(100):
#             x_val = np.random.randn(32, 1, 28, 28).astype(np.float32)
#             y_val = np.random.choice([0, 1], size=(32,)).astype(np.int64)

#             y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

#             loss_val, _ = executor.run([loss, train_op], feed_dict={
#                 x: x_val, 
#                 y: y_onehot, 
#             })

#             x_torch = torch.from_numpy(x_val)
#             y_torch = torch.from_numpy(y_val)
#             preds_torch = torch_model(x_torch)
#             loss_torch = criterion_torch(preds_torch, y_torch)
#             optimizer_torch.zero_grad()
#             loss_torch.sum().backward()
#             optimizer_torch.step()


#             diffs, ratios = [], []
#             for w1, w2 in zip(hspmd_model.parameters(), torch_model.parameters()):
#                 w1 = w1.numpy(force=True)
#                 w2 = w2.detach().numpy()
#                 assert w1.shape == w2.shape
#                 diff = np.abs(w1 - w2).sum()
#                 ratio = (np.abs((w1 - w2) / w2)).sum()
#                 diffs.append(diff)
#                 ratios.append(ratio)
#                 # print(w1, " ", w2)
#             print(diffs, ratios)
#             # print(loss_torch, " ", loss_val)

class TestResNet(unittest.TestCase):

    def test_dar_mode(self):
        weights_np = [np.random.uniform(-0.005, 0.005, size=(3, 1, 3, 3)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(3,)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(3, 3, 3, 3)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(3,)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(3, 1, 5, 5)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(3,)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(1728, 10)).astype(np.float32),
                      np.random.uniform(-0.005, 0.005, size=(10,)).astype(np.float32),]

        hspmd_model = Hetu_ResidualBlock(weights_np, 3)
        x = hspmd.placeholder(hspmd.float32, shape=[-1, 1, 28, 28])
        y = hspmd.placeholder(hspmd.float32, shape=[-1, 10])
        pred = hspmd_model(x)
        loss = hspmd.softmax_cross_entropy(pred, y)
        # probs = hspmd.sigmoid(pred)
        # loss = hspmd.binary_cross_entropy(probs, y, "mean")

        optimizer = hspmd.optim.SGD(lr=0.1)
        train_op = optimizer.minimize(loss)

        torch_model = ResidualBlock_low(weights_np, 3)
        criterion_torch = torch.nn.CrossEntropyLoss()
        optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)

        executor = hspmd.DARExecutor("cuda:0")

        for i in range(100):
            x_val = np.random.randn(32, 1, 28, 28).astype(np.float32)
            y_val = np.random.choice([0, 1], size=(32,)).astype(np.int64)

            y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

            x_torch = torch.from_numpy(x_val)
            y_torch = torch.from_numpy(y_val)
            preds_torch = torch_model(x_torch)
            loss_torch = criterion_torch(preds_torch, y_torch)
            optimizer_torch.zero_grad()
            loss_torch.sum().backward()
            optimizer_torch.step()


            loss_val, _ = executor.run([loss, train_op], feed_dict={
                x: x_val, 
                y: y_onehot, 
            })


            diffs, ratios = [], []
            for w1, w2 in zip(hspmd_model.parameters(), torch_model.parameters()):
                w1 = w1.numpy(force=True)
                w2 = w2.detach().numpy()
                assert w1.shape == w2.shape
                diff = np.abs(w1 - w2).sum()
                ratio = (np.abs((w1 - w2) / w2)).sum()
                diffs.append(diff)
                ratios.append(ratio)
                # print(w1, " ", w2)
            print(diffs, ratios)
            # print(loss_torch, " ", loss_val)

if __name__ == "__main__":
    unittest.main()
