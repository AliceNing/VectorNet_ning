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
from data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn
import pickle
root_path = os.path.dirname(os.path.abspath(__file__))

config = dict()
# Number of timesteps the track should exist to be considered in social context
config["EXIST_THRESHOLD"] = (50)
# index of the sorted velocity to look at, to call it as stationary
config["STATIONARY_THRESHOLD"] = (13)
config["color_dict"] = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
config["LANE_RADIUS"] = 30 #30 [100, -100, 100, -100]
config["DATA_DIR"] = './data'
config["OBS_LEN"] = 20
config["query_bbox"] = [-100, 100, -100, 100]
config["batch_size"] = 2
config["workers"] = 0
config["train_dir"] = "./data/train"   #os.path.join(root_path, "data/train")
config["val_dir"] = "./data/val"       #os.path.join(root_path, "./data/val")
config["test_dir"] = "./data/test"     #os.path.join(root_path, "./data/test")
# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "dataset","preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path,"dataset", "preprocess", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(root_path, "dataset", 'preprocess', 'test_test.p')

# sys.path.insert(0, root_path)


def train(config):
    # Data loader for training set
    dataset = Dataset(config["train_dir"], config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],  #加载数据进程数
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True, #拷贝数据到cuda
        drop_last=False,
    )

    stores = [None for x in range(2016)]  #205942
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)): #batch_num 0-6435
        # little dataset
        # if i > 62:
        #     break

        data = dict(data)
        for j in range(len(data["idx"])):#batch 0-31
            store = dict()
            for key in [
                "norm_center",
                "rot",
                "item_num",
                "poly_feats",
                "gt_preds",
                "has_preds",
                "city",
                "trajs",
                "timestamp",
                "step",
                "theta",
                "idx",
            ]:
                store[key] = to_numpy(data[key][j])

            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:  #print time
            print(i, time.time() - t)
            t = time.time()

    modify(config, data_loader, config["preprocess_train"])
    # write preprocessed  data
    f = open(os.path.join(root_path, 'preprocess', config["preprocess_train"]), 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

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

if __name__ == "__main__":
    config["preprocess"] = False  # we use raw data to generate preprocess data
    config["val_workers"] = 1#32
    config["workers"] = 0 #32
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi

    os.makedirs(os.path.dirname(config['preprocess_train']), exist_ok=True)

    # val(config)
    # test(config)
    train(config)
