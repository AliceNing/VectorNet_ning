# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

"""
Preprocess the data(csv), build graph from the HDMAP and saved as pkl
"""

import argparse
import os
import pickle
import random
import sys
import time
from importlib import import_module

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn
from utils import Logger, load_pretrain, gpu

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


config = dict()
config["RAW_DATA_FORMAT"] = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
config["LANE_WIDTH"] = {'MIA': 3.84, 'PIT': 3.97}
config["VELOCITY_THRESHOLD"] = 1.0
# Number of timesteps the track should exist to be considered in social context
config["EXIST_THRESHOLD"] = (50)
# index of the sorted velocity to look at, to call it as stationary
config["STATIONARY_THRESHOLD"] = (13)
config["color_dict"] = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
config["LANE_RADIUS"] = 5#30
config["OBJ_RADIUS"] = 30
config["DATA_DIR"] = './data'
config["OBS_LEN"] = 20
config["INTERMEDIATE_DATA_DIR"] = './interm_data'
config["query_bbox"] = [-100, 100, -100, 100]


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    print(args.model)
    config, *_ = model.get_model()

    config["preprocess"] = False  # we use raw data to generate preprocess data
    config["val_workers"] = 32
    config["workers"] = 32
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi

    os.makedirs(os.path.dirname(config['preprocess_train']), exist_ok=True)



    # val(config)
    # test(config)
    train(config)


def train(config):
    # Data loader for training set
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],  #加载数据进程数
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True, #拷贝数据到cuda
        drop_last=False,
    )

    stores = [None for x in range(2016)]#205942   little dataset:  63*32
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)): #batch_num 0-6435
        # little dataset
        if i > 62:
            break

        data = dict(data)
        for j in range(len(data["idx"])):#batch 0-31
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])

            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()


    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_train"])


def val(config):
    # Data loader for validation set
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(2016)]  #39472

    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        # little dataset
        if i > 62:
            break
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_val"])


def test(config):
    dataset = Dataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(1024)] #78143

    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        # little dataset
        if i > 31:
            break
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_test"])


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


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data


def modify(config, data_loader, save):
    t = time.time()
    store = data_loader.dataset.split
    for i, data in enumerate(data_loader):
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            # preprocess(data[j], config['cross_dist'])
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))  #'cross_dist': 6

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

    # write preprocessed  data  !!!
    f = open(os.path.join(root_path, 'preprocess', save), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from data import from_numpy, ref_copy
        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            # graph[key] = ref_copy(data['graph'][key])
            # Nonetype data
            try:
                graph[key] = ref_copy(data['graph'][key])
            except:
                continue
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


if __name__ == "__main__":
    main()
