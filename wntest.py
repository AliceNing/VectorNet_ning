import torch
# v_num = torch.randint(1, 10, (1,)).int()
# a = torch.rand(v_num, 9)
# b = torch.cat([a] *3, axis=0)
# print(b)
# print(v_num)

a = torch.arange(0,6)
a = a.view(2,3)
print(a)
b = a.unsqueeze(0)
print(b)
c = b.squeeze(0)
print(c)
d = b.squeeze(1)  #只有维度为1时才会去掉
print(d)
