# -*- encoding: utf-8 -*-
"""
@File    : Activations.py
@Time    : 4/10/20 22:36
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from Layers import LayerBase
import numpy as np

class ELU(LayerBase):
    def __init__(self, alpha = 1):
        super(ELU, self).__init__()
        self.params["alpha"] = alpha

    def forward(self, x):
        pos_mask = (x > 0).astype(float)
        pos_masked_x = pos_mask * x
        neg_mask = (x <= 0).astype(float)
        neg_masked_x = self.params["alpha"] * (np.exp(neg_mask * x) - 1)

        self.results["x"] = x
        self.results["pos_mask"] = pos_mask
        self.results["neg_mask"] = neg_mask
        self.results["value"] = pos_masked_x + neg_masked_x

        return self.results["value"]

    def backward(self, grads):
        value = self.results["value"]
        pos_mask = self.results["pos_mask"]
        neg_mask = self.results["neg_mask"]
        alpha = self.params["alpha"]
        pos_grads = pos_mask * grads
        neg_grads = (value * neg_mask + alpha) * grads
        cur_grads = pos_grads + neg_grads
        self.results = {}
        return cur_grads


class ReLu(LayerBase):
    def __init__(self):
        super(ReLu, self).__init__()

    def forward(self, x):
        pos_mask = (x>0).astype(float)
        masked_x = pos_mask * x
        self.results["value"] = masked_x
        self.results["x"] = x
        self.results["pos_mask"] = pos_mask
        return masked_x
    def backward(self, grads):
        pos_mask = self.results["pos_mask"]
        cur_grads = pos_mask * grads
        self.results = {}
        return cur_grads


if __name__ == '__main__':

    activation1 = ELU()
    activation2 = ReLu()

    value1 = activation1.forward(x)
    value2 = activation2.forward(x)
    print(value1)
    print(value2)

    last_grads_1 = np.array([[1,1],[-1,-2]])
    last_grads_2 = np.array([[1.2,1.3],[-1,0]])

    cur_grads_1 = activation1.backward(last_grads_1)
    cur_grads_2 = activation2.backward(last_grads_2)

    print(cur_grads_1)
    print(cur_grads_2)


