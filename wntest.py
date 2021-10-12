import torch
from torch import nn
import numpy as np
# from util import *
# v_num = torch.randint(1, 10, (1,)).int()
# a = torch.rand(v_num, 9)
# b = torch.cat([a] *3, axis=0)
# print(b)
# print(v_num)

# a = torch.arange(0,6)
# a = a.view(2,3)
# print(a)
# b = a.unsqueeze(0)
# print(b)
# c = b.squeeze(0)
# print(c)
# d = b.squeeze(1)  #只有维度为1时才会去掉
# print(d)

"""list numpy tensor 转换"""
a1 = torch.tensor([[1,2,3]])
a2 = torch.tensor([[2,3,4]])
a3 = torch.tensor([[5,4,6]])
b = [a1,a2,a3]
c = torch.stack(b, 0)
d = torch.squeeze(c, 1)
# c1 = torch.cat(b, 0)
print("asd")

"""nn.Linear"""
# m = nn.Linear(20, 30)  #  weights:20*30
# # input = torch.randn(128, 20)
# input = torch.randn(2, 128, 20)
# output = m(input)
# print(output.size())

"""nn.norm"""
# input = torch.randn(20, 5, 10, 10)
# m = nn.LayerNorm(10)
# output = m(input)
# print("sdf")

# x = torch.zeros(2, 1, 2, 1, 2)
# print(x.size())
# y = torch.squeeze(x,3)
# print(y.size())
