import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from config import *
import pandas as pd
from typing import List, Dict, Any
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from torch.utils.data import DataLoader
from skimage.transform import rotate


class ArgoDataset(Dataset):
    def __init__(self, dir, config, train=True, pad = False):
        self.config = config
        self.train = train
        self.pad = pad

        if 'preprocess' in config and config['preprocess']:#加载预处理好的数据
            if train:
                self.load_file = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.load_file = np.load(self.config['preprocess_val'], allow_pickle=True)
        else:#第一次数据预处理
            self.avl = ArgoverseForecastingLoader(dir)
            self.avl.seq_list = sorted(self.avl.seq_list)
            self.am = ArgoverseMap()
            
    def __getitem__(self, idx):

        '''lanegcn'''
        #加载处理好的数据
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.load_file[idx]
            return data

        #第一次数据预处理
        '''data:  
        X: xs,ys,xe,ye,type,att1,att2,att3,id
        y: future step, pre-step,  time-step,2
        ROT:
        NORM_CENTER:
        '''
        data = self.read_agt_obj_data(idx)
        data['idx'] = idx
        data = self.get_obj_feats(data)
        data = self.read_lane_data(data)
        data = self.filter_data(data) #筛选特征
        return data
    
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.load_file)
        else:
            return len(self.avl)

    def read_agt_obj_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)
        traj_df = copy.deepcopy(self.avl[idx].seq_df)
        # traj_df: TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME

        '时间戳归一化'
        traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
        traj_timestap = np.sort(np.unique(traj_df['TIMESTAMP'].values))
        traj_len = traj_timestap.shape[0]
        mapping = dict()
        for i, ts in enumerate(traj_timestap):
            mapping[ts] = i
        steps = [mapping[x] for x in traj_df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        'agent traj'
        all_trajs = np.concatenate((
            traj_df.X.to_numpy().reshape(-1, 1),
            traj_df.Y.to_numpy().reshape(-1, 1)), 1)
        all_ts = traj_df.TIMESTAMP.to_numpy()

        all_objs = traj_df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        # 统计不同的object和其唯一ID（Keys，0:id,1:type），索引（value）
        keys = list(all_objs.keys())
        obj_type = [x[1] for x in keys]
        agt_idx = obj_type.index('AGENT')
        idcs = all_objs[keys[agt_idx]]

        # 拿到所有的AGENT
        agt_traj = all_trajs[idcs]
        agt_timestamp = all_ts[idcs]
        agt_step = steps[idcs]

        # 重新按时间排序
        agt_idcs = agt_step.argsort()
        agt_step = agt_step[agt_idcs]
        agt_traj = agt_traj[agt_idcs]
        agt_timestamp = agt_timestamp[agt_idcs]

        'agent 的上下文ctx_obj traj'
        del keys[agt_idx]
        ctx_trajs, ctx_timestamps, ctx_steps = [], [], []
        for key in keys:  # keys[('id',type)...]
            idcs = all_objs[key]
            ctx_traj = all_trajs[idcs]
            ctx_timestamp = all_ts[idcs]
            ctx_step = steps[idcs]
            # 重新按时间排序
            ctx_idcs = ctx_step.argsort()
            ctx_step = ctx_step[ctx_idcs]
            ctx_traj = ctx_traj[ctx_idcs]
            ctx_timestamp = ctx_timestamp[ctx_idcs]

            ctx_trajs.append(ctx_traj)
            ctx_timestamps.append(ctx_timestamp)
            ctx_steps.append(ctx_step)


        '旋转矩阵'
        norm_center = agt_traj[19].copy().astype(np.float32)
        # AGENT的第20个时刻位置做为norm_center
        pre = agt_traj[18] - norm_center
        theta = np.pi - np.arctan2(pre[1], pre[0])
        rot = torch.tensor(np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32))

        data_tmp = dict()
        data_tmp['city'] = city
        # AGENT对象放在第一个，其余是其上下文
        data_tmp['trajs'] = [agt_traj] + ctx_trajs  # obj_num, step_num, 2
        data_tmp['timestamp'] = [agt_timestamp] + ctx_timestamps
        data_tmp['step'] = [agt_step] + ctx_steps
        data_tmp['norm_center'] = norm_center
        data_tmp['rot'] = rot
        data_tmp['theta'] = theta
        return data_tmp

    def get_obj_feats(self, data):
        poly_feats = [] #, gt_preds, has_preds
        type_flag = True
        id = torch.tensor([0], dtype = torch.int32)
        for traj, step ,timestamp in zip(data['trajs'], data['step'], data['timestamp']):
            if 19 not in step:  # 删掉step不够20的数据
                continue

            '''处理预测数据   step不够的补0，has_pred = 0表示补0的
            per-step offsets, 从norm-center开始
            '''
            gt_pred = torch.tensor(np.zeros((30, 2), np.float32))
            has_pred = torch.tensor(np.zeros(30, np.bool))
            future_mask = np.logical_and(step >= 20, step < 50)

            post_step = step[future_mask] - 20# future step 从0开始计数
            post_traj = traj[future_mask] - data['norm_center']

            '''处理观察到的数据
            speed   变成vector
            '''
            obs_mask = step < 20
            obs_step = step[obs_mask]
            obs_traj = traj[obs_mask]
            obs_timestamp = timestamp[obs_mask]

            for i in range(len(obs_step)):
                temp = 19 - (len(obs_step) - 1) + i
                if obs_step[i] == 19 - (len(obs_step) - 1) + i:
                    break
            obs_step = obs_step[i:]
            obs_traj = obs_traj[i:]
            #做数据统一的处理

            # 特征处理成9维的，xs,ys,xe,ye,type,att1,att2,att3,id
            poly_feat = torch.tensor(np.zeros((19, 9), np.float32))

            feat = np.zeros((20, 2), np.float32)
            feat[obs_step] = np.matmul(data['rot'], (obs_traj - data['norm_center'].reshape(-1, 2)).T).T
            if len(obs_step)!=len(obs_timestamp):
                continue

            ts = torch.tensor(np.zeros((20), np.float32))
            ts[obs_step] = torch.tensor(obs_timestamp, dtype=torch.float)

            # 删掉超范围的
            x_min, x_max, y_min, y_max = self.config['query_bbox']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            type = 1
            if type_flag:
                type = 0 #只有第一次agent 时为0，其他obj为1
                type_flag = False
                gt_pred[post_step] = torch.tensor(post_traj,dtype=torch.float)
                has_pred[post_step] = 1
                # gt_preds.append(gt_pred)
                # has_preds.append(has_pred)

            #speed
            vector = torch.tensor(np.hstack((feat[:-1], feat[1:])))
            speed = torch.tensor([np.sqrt(x ** 2 + y ** 2) for x, y in zip((vector[:,0]-vector[:,2]), (vector[:,1]-vector[:,3]))])

            poly_feat[:,0:4] = vector  #xs,ys,xe,ye
            poly_feat[:,4] = type  # type: agent 0, obj 1, lane 2
            poly_feat[:,5] = ts[:-1]  # att1 start time
            poly_feat[:,6] = ts[1 :]  # att2 end time
            poly_feat[:,7] = speed  # att3 speed
            poly_feat[:,8] = id[0]  # id

            id[0] += 1
            poly_feats.append(poly_feat)

        # poly_feats = np.asarray(poly_feats, np.float32)
        # gt_preds = np.asarray(gt_preds, np.float32)
        # has_preds = np.asarray(has_preds, np.bool)

        data['poly_feats'] = poly_feats
        data['gt_preds'] = gt_pred
        data['has_preds'] = has_pred
        data['item_num'] = id
        return data

    def read_lane_data(self, data):
        id = data['item_num']
        x_min, x_max, y_min, y_max = self.config['query_bbox']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['norm_center'][0], data['norm_center'][1], data['city'],radius)
        for lane_id in lane_ids:
            traffic_control = self.am.lane_has_traffic_control_measure(
                lane_id, data['city'])
            is_intersection = self.am.lane_is_in_intersection(lane_id, data['city'])
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            centerlane = self.am.get_lane_segment_centerline(lane_id, data['city'])  # 10,3维 包括高度
            # normalize to last observed timestamp point of agent and rot
            centerlane[:, :2] = np.matmul(data['rot'], (centerlane[:, :2] - data['norm_center']).T).T

            """得到lane的左右车道线坐标  调用现成的方法"""
            pts = self.am.get_lane_segment_polygon(lane_id, data['city'])
            pts_len = (pts.shape[0] - 1) // 2
            # if pts_len != 10:
            #     print(pts_len)
            lane_1 = np.matmul(data['rot'], (pts[:pts_len, 0:2] - data['norm_center']).T).T
            lane_2 = np.matmul(data['rot'], (pts[pts_len:2 * pts_len, 0:2] - data['norm_center']).T).T

            if self.pad:
                #vector长度都为19，pad补0
                poly_feat = torch.tensor(np.zeros((19, 9), np.float32))
                poly_feat1 = torch.tensor(np.zeros((19, 9), np.float32))
            else:
                #vector长度不一样
                poly_feat = torch.tensor(np.zeros((pts_len-1, 9), np.float32))
                poly_feat1 = torch.tensor(np.zeros((pts_len-1, 9), np.float32))

            # lane1_feature:  xs,ys,xe,ye,type,att1,att2,att3,id
            poly_feat[:pts_len-1, 0:4] = torch.tensor(np.hstack((lane_1[0:(pts_len - 1)], lane_1[1:])))  # xs,ys,xe,ye
            poly_feat[:pts_len-1, 4] = 2  # type: agent 0, obj 1, lane 2
            if traffic_control:
                poly_feat[:pts_len-1, 5] = 1  # att1 traffic_control
            else:
                poly_feat[:pts_len-1, 5] = 0
            if lane.turn_direction == 'LEFT':  # att2 direction
                poly_feat[:pts_len-1, 6] = 1
            elif lane.turn_direction == 'RIGHT':
                poly_feat[:pts_len-1, 6] = 2
            else:
                poly_feat[:pts_len-1, 6] = 0
            if is_intersection:
                poly_feat[:pts_len-1, 7] = 1  # att3 intersection
            else:
                poly_feat[:pts_len-1, 7] = 0
            poly_feat[:pts_len-1, 8] = id[0]  # id

            data['poly_feats'].append(poly_feat)
            id[0] +=1

            # lane2_feature:  xs,ys,xe,ye,type,att1,att2,att3,id
            poly_feat1[:pts_len-1, 0:4] = torch.tensor(np.hstack((lane_2[0:(pts_len - 1)], lane_2[1:])))  # xs,ys,xe,ye
            poly_feat1[:pts_len-1, 4] = 2  # type: agent 0, obj 1, lane 2
            if traffic_control:
                poly_feat1[:pts_len-1, 5] = 1  # att1 traffic_control
            else:
                poly_feat1[:pts_len-1, 5] = 0
            if lane.turn_direction == 'LEFT':  # att2 direction
                poly_feat1[:pts_len-1, 6] = 1
            elif lane.turn_direction == 'RIGHT':
                poly_feat1[:pts_len-1, 6] = 2
            else:
                poly_feat1[:pts_len-1, 6] = 0
            if is_intersection:
                poly_feat1[:pts_len-1, 7] = 1  # att3 intersection
            else:
                poly_feat1[:pts_len-1, 7] = 0
            poly_feat1[:pts_len-1, 8] = id[0]  # id

            data['poly_feats'].append(poly_feat1)
            id[0] += 1

        data['item_num'] = id
        # data['poly_feats'] = np.asarray(data['poly_feats'], np.float32)
        return data

    def filter_data(self, data):
        '''new_data:
        item_num:  poly_line 数量
        polyline_list:  所有的折线的向量特征  vector，注意：AGENT总为第一个
        gt_preds：  per-step  ，相对最后观测点的位移  (future)
        rot： 旋转矩阵
        '''
        new_data = dict()
        new_data['item_num'] = data['item_num']
        new_data['polyline_list'] = data['poly_feats']
        new_data['rot'] = data['rot']
        new_data['gt_preds'] = data['gt_preds']
        new_data['has_preds'] = data['has_preds']
        new_data['idx'] = data['idx']

        return new_data

def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

def collate_fn(batch):  #用list方式加载
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


if __name__ == "__main__":
    dataset = ArgoDataset("./data/train", config, train=True, pad=True)  # "dataset/train/data"
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], drop_last=True,)
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            print("epoch", epoch, "的第", i, "个inputs",
                  "item_num:", data["item_num"][0].data.size(),
                  "polyline_list:", len(data["polyline_list"][0]),
                  "future:", data["gt_preds"][0].data.size())
            # if i == 99:
            #     print("1")

