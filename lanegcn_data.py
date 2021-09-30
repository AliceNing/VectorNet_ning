# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate


class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train

            
    def __getitem__(self, idx):

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx

        data['graph'] = self.get_lane_graph(data)
        return data
    
    def __len__(self):
        return len(self.avl)

    #get city, traj, step of obj
    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        df = copy.deepcopy(self.avl[idx].seq_df)
        #df: TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME
        #一个argo数据里有多个object,且有重复，eg:794，只有70种obj
        
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        #时间戳唯一化，并排序

        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i
        #时间戳序列化, {time:0,time1:1}

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)
        #时间戳变成一个序列的形式，[0,0..1..1,2..2……49..49]

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        #统计不同的object和其唯一ID（Keys，0:id,1:type），索引（value）
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
        #拿到所有的AGENT
       
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:#keys[('id',type)...]
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
        #拿到AGENT的上下文

        data = dict()
        data['city'] = city
        #AGENT对象放在第一个，其余是其上下文
        data['trajs'] = [agt_traj] + ctx_trajs  #obj_num, step_num, 2
        data['steps'] = [agt_step] + ctx_steps  #obj_num, step_num
        return data

    #get feats, ctr, orig, gt_pred of obj
    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(np.float32)
        #AGENT的第20个时刻位置做为Orig

        if self.train and self.config['rot_aug']:#false
            theta = np.random.rand() * np.pi * 2.0
        else:
            #计算旋转角和旋转矩阵
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step: #删掉step不够20的数据
                continue

            #step不够的补0，has_pred = 0表示补0的
            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)

            # if len(step) < 50:
            #     print("not enough")
            # time 0-29 :train    20-49:test
            future_mask = np.logical_and(step >= 20, step < 50)
            # 处理预测数据
            post_step = step[future_mask] - 20
            # future step 从0开始计数
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1

            #
            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()#返回排序后的索引
            # 重新按时间排了下序
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                temp = 19 - (len(step) - 1) + i
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            #做数据统一的处理

            # 特征处理成三维的，前两位是位置特征进行处理后的坐标，第三位表示是否为pad
            feat = np.zeros((20, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            #删掉超范围的
            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())  #20_step的位置
            feat[1:, :2] -= feat[:-1, :2]  #相邻时刻的时间差
            feat[step[0], :2] = 0  # 第0step的位置为0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        # lane_ids是一部分道路的ID，有10个lane_node位置
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)#Orig就是AGENT的第20个时刻位置
        #get_lane_ids_in_xy_bbox:Get all lane IDs with a Manhattan distance search radius in the xy plane.
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            # centerline有十对位置,一个lane有9个lane_node.
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            # 进行旋转变换
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        #对每一个lane_seg做处理
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id] #一个lane_seg
            ctrln = lane.centerline
            #ctrln: has num_seg+1 xy,
            num_segs = len(ctrln) - 1

            # t = ctrln[:-1]
            # t1 = ctrln[1:]
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))  #旋转变换之后的中心位置(俩位置的中心点)
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))  #旋转变换之后的位移差

            # 两个标志位表示方向
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':  #10
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':  #01
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
        #ctrs, feats, turn, control, intersect over!
        #ctrs: lane_seg_num,9,2   lane_seg_num equal to len(lane_ids)
        #feats: lane_seg_num,9,2
        #turn:  lane_seg_num,9,2
        #control: lane_seg_num,9
        #intersect: lane_seg_num,9
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            # 记录所有lane_seg里all_lane_node的id.
            #node_idcs = [range(0,9),range(9,18)....]
            count += len(ctr)
        num_nodes = count  #all_node数量。

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)
        # 类似于[0,0,0,0,0,0,0,0,0,1X9,2X9,…………]
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:  #u和v对应csr_matrix矩阵存储方式中的行raw和列col, 不按顺序也没关系，只要行列对应就行
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]  #lane_seg
            idcs = node_idcs[i]    #lane_seg里的lane_node_id，eg:[0-8]
            
            pre['u'] += idcs[1:]   #'u':1-8  行
            pre['v'] += idcs[:-1]  #'v':0-7  列
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:  #前驱lane_seg 多个
                    if nbr_id in lane_ids:        #前驱lane_seg_id
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])  #补充第一个lane_node
                        pre['v'].append(node_idcs[j][-1])  #补充前驱lane_seg的最后一个lane_node
                    
            suc['u'] += idcs[:-1]   #'u':0-7
            suc['v'] += idcs[1:]    #'v':1-8
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])  #补充最后一个lane_node
                        suc['v'].append(node_idcs[j][0])   #补充后继lane_seg的第一个lane_node

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        #存的是相关的lane_seg_index
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])  #两个lane_seg的索引id对

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)
                    
        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs   #lanesegment_index0,  lanesegment_index1
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            if 'scales' in self.config and self.config['scales']:#false
                graph[key] += dilated_nbrs2(graph[key][0], graph['num_nodes'], self.config['scales'])
            else:
                graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        return graph #返回的pre和suc中保存了6个矩阵


class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):

        self.config = config
        self.train = train
        split2 = config['val_split'] if split=='val' else config['test_split']
        split = self.config['preprocess_val'] if split=='val' else self.config['preprocess_test']

        self.avl = ArgoverseForecastingLoader(split2)
        self.avl.seq_list = sorted(self.avl.seq_list)
            

    def __getitem__(self, idx):

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data
    
    def __len__(self):
        return len(self.avl)

