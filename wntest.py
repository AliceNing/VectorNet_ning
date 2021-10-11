import torch
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

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.cuda()
    return data.contiguous().cuda()

"""list numpy tensor 转换"""
a = [[1,2,3], [2,3,4,5,6], [5,4,6,2]]
b = gpu(a)
d = np.array(a, dtype=np.float32)
# b = b.astype(float)
c =  torch.from_numpy(b)
print("asdff")

