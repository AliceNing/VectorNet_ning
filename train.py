import os
import torch
import argparse
import numpy as np
import random
from util import *
import sys
import time
import shutil
from tqdm import tqdm
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
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./results/lanegcn_parallel')

os.umask(0) #  #修改文件模式，让进程有较大权限，保证进程有读写执行权限，这个不是一个好的方法。
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

def main():  # train on single gpu
    vector_net = VectorNetWithPredicting(v_len=9, time_stamp_number=30)
    Dataset = ArgoDataset
    learning_rate = 0.001
    decayed_factor = 0.3
    opt = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        vector_net = nn.DataParallel(vector_net)
    vector_net.to(config['device'])

    """save model and Resume"""
    #To do

    """Data loader for training and eval"""
    train_dataset = Dataset(config["train_dir"], config, set='train', pad=True)  #"dataset/train/data"
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["workers"],drop_last=True,)

    eval_dataset = Dataset(config["val_dir"], config, set='val', pad=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], num_workers=config["workers"],drop_last=True,)

    epoch = 25
    train_model(epoch, train_loader, eval_loader, vector_net, opt)
    # val_model(eval_loader, vector_net)

def train_model(epochs, train_loader, eval_loader, vector_net, optimizer, learning_rate=0.001, decayed_factor=0.3):
    vector_net.train()
    iter_num=0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            data = dict(data)
            optimizer.zero_grad()
            outputs = vector_net(data)
            loss = loss_func(outputs, data['gt_preds'])
            loss.backward()
            optimizer.step()
            'tensorboard add scaler every iter'
            ade = torch.mean(get_ADE(outputs, data["gt_preds"])).cpu().detach().numpy()
            writer.add_scalar("ade", ade, iter_num)
            fde = torch.mean(get_FDE(outputs, data["gt_preds"])).cpu().detach().numpy()
            writer.add_scalar("fde", fde, iter_num)
            de1 = torch.mean(get_DE(outputs, data["gt_preds"], [9])).cpu().detach().numpy()
            writer.add_scalar("de1", de1, iter_num)
            de2 = torch.mean(get_DE(outputs, data["gt_preds"], [19])).cpu().detach().numpy()
            writer.add_scalar("de2", de2, iter_num)
            de3 = torch.mean(get_DE(outputs, data["gt_preds"], [29])).cpu().detach().numpy()
            writer.add_scalar("de3", de3, iter_num)
            iter_num += 1
            'learning rate decay'
            if (epoch + 1) % 5 == 0:
                learning_rate *= decayed_factor
                optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
            'val after 2 epoch'
            if epoch !=0 and epoch % 1 ==0:
                val_model(eval_loader, vector_net)
            if i %10==0:
                print("epoch:", epoch, "iteration:", i, "loss function:", loss.item())
        save_ckpt(vector_net, optimizer, config, epoch)

def val_model(eval_loader, vector_net):
    vector_net.eval()  #固定norm和drop
    loss, ade, fde, de = 0, 0, 0, 0
    for i, data in enumerate(eval_loader):
        data = dict(data)
        outputs = vector_net(data)
        loss += loss_func(outputs, data['gt_preds'])
        ade += torch.mean(get_ADE(outputs, data["gt_preds"]))
        fde += torch.mean(get_FDE(outputs, data["gt_preds"]))
        de += torch.mean(get_DE(outputs, data["gt_preds"], [9,19,29]))
    print("Metrics on eval dataset:", "loss:", loss.item(), "ADE:", ade.item(), "FDE:", fde.item(), "DE:", de.item())
    vector_net.train()

def worker_init_fn(pid):
    np_seed = 0 * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

def save_ckpt(net, opt, config, epoch):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

# net.state_dict存放训练过程中需要学习的权重和偏执系数,只包含卷积层和全连接层的参数
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.state_dict()},
        os.path.join(config['save_dir'], save_name),
    )

if __name__ == "__main__":
    main()
