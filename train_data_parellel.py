import os
os.umask(0) #  #修改文件模式，让进程有较大权限，保证进程有读写执行权限，这个不是一个好的方法。 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
from util import *
import sys
import time
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Sampler, DataLoader
from numbers import Number
from network.vector_net import VectorNet, VectorNetWithPredicting
from dataProcess.data import *
from loss_and_eval.evaluation import *
from loss_and_eval.loss import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

def main():
    vector_net = VectorNetWithPredicting(v_len=9, time_stamp_number=30)
    Dataset = ArgoDataset
    learning_rate = 0.001
    decayed_factor = 0.3
    # Adam without clip
    opt = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        vector_net = nn.DataParallel(vector_net)
    vector_net.to(device)

    # Data loader for training and eval
    train_dataset = Dataset(config["train_dir"], config, train=True)  #"dataset/train/data"
    # train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["workers"],
                              pin_memory=True, collate_fn=collate_fn, worker_init_fn=worker_init_fn, drop_last=True,)

    eval_dataset = Dataset(config["val_dir"], config, train=False)
    # eval_sampler = DistributedSampler(eval_dataset, num_replicas=1, rank=0)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], num_workers=config["workers"],
                             pin_memory=True, collate_fn=collate_fn, worker_init_fn=worker_init_fn, drop_last=True,)

    epoch = 25
    train_model(epoch, train_loader, eval_loader, vector_net, opt)

def train_model(epochs, train_loader, eval_loader, vector_net, optimizer, is_print_eval=True,  is_print=True,  learning_rate=0.001, decayed_factor=0.3):
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            data = dict(data)
            optimizer.zero_grad()
            # outputs = vector_net(data["item_num"][0].to(config['device']), data["polyline_list"])
            outputs = vector_net(data)
            loss = loss_func(outputs, data['gt_preds'])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                learning_rate *= decayed_factor
                optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)

            if is_print:
                print("epoch:", epoch, "iteration:", i, "loss function:", loss.item())

            if is_print_eval:
                loss, ade, t = 0, 0, 0
                for i, data in enumerate(eval_loader):
                    data = dict(data)
                    outputs = vector_net(data)
                    loss += loss_func(outputs, data['gt_preds'])
                    ade += torch.mean(get_ADE(outputs, data["gt_preds"]))
                    t += 1
                if t > 0:
                    loss /= t
                    ade /= t
                    print("epoch:", epoch, "Mean metrics on eval dataset:", "loss:", loss, "ADE:", ade)
            # print("asdfafa")
    save_name = "epoch" + str(epoch)
    torch.save(vector_net, os.path.join(config.model_save_path, save_name + '.model'))

def worker_init_fn(pid):
    np_seed = 0 * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

if __name__ == "__main__":
    main()
