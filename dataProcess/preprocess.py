from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from tqdm import tqdm
import re
import time
import sys
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import pickle
from data import ArgoDataset as Dataset
from pathlib import Path

path = Path(__file__).parents[0]
root_path = os.path.dirname(path)
# sys.path.append(root_path)
sys.path.insert(0, root_path)
from config import *

def train(config):
    # Data loader for training set
    dataset = Dataset(config["train_dir"], config, 'train', pad = True)
    train_loader = DataLoader(dataset, batch_size=config["preprocess_batch_size"], num_workers=config["workers"],  #加载数据进程数
        shuffle=False, drop_last=False,)
    stores = [None for x in range(205942)]  #205942,200
    for i, data in enumerate(tqdm(train_loader)): #batch_num 0-6435
        # little datase
        # if i >= 200:
        #     break
        for j in range(len(data["idx"])):#batch 0-31
            store = dict()
            for key in ['item_num', 'rot', 'gt_preds', 'has_preds', 'idx', ]:
                store[key] = data[key][j]
            store['polyline_list'] = data['polyline_list']
            stores[store["idx"]] = store
    # write preprocessed  data
    f = open(config["preprocess_train"], 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def val(config):
    # Data loader for validation set
    dataset = Dataset(config["val_dir"], config, 'val', pad = True)
    val_loader = DataLoader(dataset, batch_size=config["preprocess_batch_size"], num_workers=config["val_workers"],
        shuffle=False, )
    stores = [None for x in range(39472)]  #39472
    for i, data in enumerate(tqdm(val_loader)):  # batch_num 0-6435
        # little dataset
        # if i >= 200:
        #     break
        for j in range(len(data["idx"])):  # batch 0-31
            store = dict()
            for key in [ 'item_num', 'rot', 'gt_preds', 'has_preds', 'idx']:
                store[key] = data[key][j]
            store['polyline_list'] = data['polyline_list']
            stores[store["idx"]] = store
    # write preprocessed  data
    f = open(config["preprocess_val"], 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("preprocess val write done!!")
    f.close()

def test(config):
    dataset = Dataset(config["test_dir"], config, 'test', pad = True)
    test_loader = DataLoader(
        dataset,
        batch_size=config["preprocess_batch_size"],
        num_workers=config["val_workers"],
        # collate_fn=collate_fn,
        shuffle=False,
        # pin_memory=True,
    )
    stores = [None for x in range(78143)] #78143 ,1024

    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):  # batch_num 0-6435
        # little dataset
        # if i >= 100:
        #     break
        for j in range(len(data["idx"])):  # batch 0-31
            store = dict()
            for key in [
                'item_num',
                'rot',
                'gt_preds',
                'has_preds',
                'idx'
            ]:
                store[key] = data[key][j]
            store['polyline_list'] = data['polyline_list']


            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:  # print time
            print(i, time.time() - t)
            t = time.time()

    # write preprocessed  data
    f = open(config["preprocess_test"], 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    print("preprocess val write done!!")

def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

if __name__ == "__main__":
    config["preprocess"] = False  # we use raw data to generate preprocess data
    os.makedirs(os.path.dirname(config['preprocess_train']), exist_ok=True)

    # train(config)
    val(config)
    test(config)

