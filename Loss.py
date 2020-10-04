# -*- encoding: utf-8 -*-
"""
@File    : Loss.py
@Time    : 5/10/20 00:35
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from Layers import LayerBase
import numpy as np

class MeanSquareError(LayerBase):
    pass

class BinaryCrossEntropy(LayerBase):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, logit, labels):
        # batch_size = logit.shape[0]
        loss = -(labels * np.log(logit) + (1-labels) * np.log(1-logit))
        self.results["labels"] = labels
        self.results["logit"] = logit
        return np.sum(loss, 0)

    def backward(self):
        labels = self.results["labels"]
        logit = self.results["logit"]
        grad_logit = -np.sum(labels/logit - (1-labels)/(1-logit), 0)
        self.results = {}
        return grad_logit


# class CrossEntropy(LayerBase):
#     def __init__(self):
#         super(CrossEntropy, self).__init__()
#
#     def forward(self, x, labels):
#         # x (batch_size, 1)
#         # labels (batch_size,1)
#         batch_size = x.shape[0]
#         loss = -(labels * np.log(x) + (1-labels) * np.log(1 - x) ) # loss for each sample in batch (batch_size, 1)
#         self.results["loss"] = loss
#         self.results["x"] = x
#         self.results["labels"] = labels
#         return (-1 / batch_size) * np.sum(loss, 0)
#
#     def backward(self):
#         logit = self.results["logit"]
#         labels = self.results["labels"]
#         grads = (logit - labels)

