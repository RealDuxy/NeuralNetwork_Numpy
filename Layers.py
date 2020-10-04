# -*- encoding: utf-8 -*-
"""
@File    : Layers.py
@Time    : 4/10/20 15:27
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

import numpy as np

class LayerBase(object):
    def __init__(self, *args, **kwargs):
        self.params = {}
        self.grads = {}
        self.results = {}

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass


class Linear(LayerBase):
    def __init__(self, input_dim, output_dim, bias=False):
        super(Linear, self).__init__()
        self.W = np.random.normal(size=(input_dim, output_dim))   # N(0,1) shape(128,1)
        if bias:
            self.b = np.random.normal(size=output_dim)  # shape(4)
        else:
            self.b = []

    def forward(self, x):
        # x shape: (batch_size, input_dim) (64,128)
        z = np.dot(x, self.W)  # z shape after dot: (batch_size, output_dim)
        if self.b:
            z += np.tile(self.b, x.shape[0])
        # 记录下本l层的 输入x 和 线性映射z
        self.results["z"] = z
        self.results["x"] = x
        return z

    def backward(self, last_grads):
        # last_grads shape: (batch_size, output_dim)     \partial {L} / \partial {z^l+1}
        cur_grads = np.dot(last_grads, self.W.T)  # cur_grad after dot: (batch_size, input_dim)
        cur_x = self.results["x"].T  # (input_dim, batch_size)
        w_grad = np.dot(cur_x, last_grads)  # w_grad: (input_dim, output_dim)
        b_grad = last_grads  # b_grad: (batch_size, output_dim)
        self.grads["W"] = w_grad
        self.grads["b"] = b_grad
        self.results = {}
        return cur_grads


if __name__ == '__main__':
    network = Linear(128, 8)
    x = np.random.random((64,128))
    last_grads = np.random.random((64, 8))
    f = network.forward(x)
    b = network.backward(last_grads)





