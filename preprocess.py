from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from data import *
from tqdm import tqdm
import re
import pickle

config = dict()
config[RAW_DATA_FORMAT] = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
config[LANE_WIDTH] = {'MIA': 3.84, 'PIT': 3.97}
config[VELOCITY_THRESHOLD] = 1.0
# Number of timesteps the track should exist to be considered in social context
config[EXIST_THRESHOLD] = (50)
# index of the sorted velocity to look at, to call it as stationary
config[STATIONARY_THRESHOLD] = (13)
config[color_dict] = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
config[LANE_RADIUS] = 30
config[OBJ_RADIUS] = 30
config[DATA_DIR] = './data'
config[OBS_LEN] = 20
config[INTERMEDIATE_DATA_DIR] = './interm_data'


def compute_feature_for_one_seq(traj_df: pd.DataFrame, am: ArgoverseMap, obs_len: int = 20, lane_radius: int = 5,
                                obj_radius: int = 10, viz: bool = False, mode='rect', folder_name="",
                                query_bbox=[-100, 100, -100, 100]) -> List[List]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center np.ndarray: (2, )
    """
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    seq_len = seq_ts.shape[0]
    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [None] * 7  # 起点，终点和中间20个时刻的观察点
    # agent traj & its start/end point
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):  # 只按照类型保存？不考虑track_id???
        if obj_type == 'AGENT':  # AGENT对象的
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len - 1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # prune points after "obs_len" timestamp 把后30个时间戳的数据删掉，只留前20个时刻的数据
    # [FIXED] test set length is only `obs_len` obj的traj要保存50个时刻的还是前20个时刻的？？-ning
    if folder_name == "test":
        traj_df = traj_df[traj_df['TIMESTAMP'] <= agent_df['TIMESTAMP'].values[obs_len - 1]]

        assert (np.unique(traj_df["TIMESTAMP"].values).shape[0]
                == obs_len), "Obs len mismatch"

    """search nearby lane from the last observed point of agent
    #lane_radius=30 norm_center是last_obs的坐标 query_bbox=100*4
    #返回值:list长度是lane_num,每个里面包含一个长度为5的list:(lane_feature_ls =[halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id],)
    # 其中左右车道线坐标是三维的，包括高度信息（前两维减去norm_center了）
    # FIXME: nearby or rect?
    """
    lane_feature_ls = get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)
    # lane_feature_ls = get_nearby_lane_feature_ls(
    #     am, agent_df, obs_len, city_name, lane_radius, norm_center)

    # pdb.set_trace()

    """search nearby moving objects from the last observed point of agent
    return: xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id
    其中xys是vector（减去norm_center了）,obj类型，ts是vector两点的平均时间戳，track_id
    xys和ts的长度都是49！！
    """
    obj_feature_ls = get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, norm_center)

    """get agent features
    return:list，其包含xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0], gt_xys
    其中xys是前20个时刻的vector（减去norm_center了），ts是vector两点的平均 时间戳，gt_xys是后30个时刻的位置（减去norm_center了）
    """
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)

    # vis
    if viz:
        for features in lane_feature_ls:
            show_doubled_lane(
                np.vstack((features[0][:, :2], features[0][-1, 3:5])))
            show_doubled_lane(
                np.vstack((features[1][:, :2], features[1][-1, 3:5])))
        for features in obj_feature_ls:
            show_traj(
                np.vstack((features[0][:, :2], features[0][-1, 2:])), features[1])
        show_traj(np.vstack(
            (agent_feature[0][:, :2], agent_feature[0][-1, 2:])), agent_feature[1])

        plt.plot(agent_x_end - query_x, agent_y_end - query_y, 'o',
                 color=color_dict['AGENT'], markersize=7)
        plt.plot(0, 0, 'x', color='blue', markersize=4)
        plt.plot(start_x - query_x, start_y - query_y,
                 'x', color='blue', markersize=4)
        plt.show()

    return [agent_feature, obj_feature_ls, lane_feature_ls, norm_center]


def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets,
    starting from the last observed location.We rotate the coordinate system
    based on the heading of the target vehicle at the last observed location.

    """
    assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
    # import pdb
    # pdb.set_trace()
    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) - gt).sum()}"

    return offset_gt


def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls):
    """
    args:
        agent_feature_ls:
            list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    returns:
        pd.DataFrame of (
            polyline_features: vstack[
                (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),
                (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
                ]
            offset_gt: incremental offset from agent's last obseved point,
            traj_id2mask: Dict[int, int]
            lane_id2mask: Dict[int, int]
        )
        where obejct_type = {0 - others, 1 - agent}

    """
    polyline_id = 0  # 每个scence里polyline的ID标识
    traj_id2mask, lane_id2mask = {}, {}
    gt = agent_feature[-1]  # ground_truth
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 7))  # ??

    # encoding agent feature
    pre_traj_len = traj_nd.shape[0]
    agent_len = agent_feature[0].shape[0]  # agent obs vector len
    # print(agent_feature[0].shape, np.ones(
    # (agent_len, 1)).shape, agent_feature[2].shape, (np.ones((agent_len, 1)) * polyline_id).shape)
    agent_nd = np.hstack((agent_feature[0], np.ones(
        (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    # agent_obseve_seq4,one1,time1,polyline_id1=7维
    assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((traj_nd, agent_nd))
    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1

    # encoding obj feature
    for obj_feature in obj_feature_ls:
        # 传进来的数据长度是49，只留前19个
        obj_feature[0] = obj_feature[0][:19]
        obj_feature[2] = obj_feature[2][:19]

        obj_len = obj_feature[0].shape[0]
        assert obj_feature[2].shape[0] == obj_len, f"obs_len of obj is {obj_len}"
        if not obj_feature[2].shape[0] == obj_len:
            from pdb import set_trace;
            set_trace()
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))

        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]  # 车道左边缘线
        l_lane_nd = np.hstack(
            (lane_feature[0], np.ones((l_lane_len, 1)) * polyline_id))  # 6个坐标，加一个ID位
        assert l_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]  # 车道右边缘线
        r_lane_nd = np.hstack(
            (lane_feature[1], np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

    # FIXME: handling `nan` in lane_nd
    col_mean = np.nanmean(lane_nd, axis=0)
    if np.isnan(col_mean).any():
        # raise ValueError(
        # print(f"{col_mean}\nall z (height) coordinates are `nan`!!!!")
        lane_nd[:, 2].fill(.0)
        lane_nd[:, 5].fill(.0)
    else:
        inds = np.where(np.isnan(lane_nd))
        lane_nd[inds] = np.take(col_mean, inds[1])

    # traj_ls, lane_ls = reconstract_polyline(
    #     np.vstack((traj_nd, lane_nd)), traj_id2mask, lane_id2mask, traj_nd.shape[0])
    # type_ = 'AGENT'
    # for traj in traj_ls:
    #     show_traj(traj, type_)
    #     type_ = 'OTHERS'

    # for lane in lane_ls:
    #     show_doubled_lane(lane)
    # plt.show()

    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(gt)
    """faeture 改变顺序"""
    # change lanes feature from (xs, ys, zs, xe, ye, ze, polyline_id) to (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
    lane_nd = np.hstack([lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])  # timestap???
    lane_nd = lane_nd[:, [0, 1, 3, 4, 7, 2, 5, 6]]  # x1,y1,x2,y2,0,z1,z2,polyline_id

    # change object features from (xs0, ys1, xe2, ye3, obejct_type4, timestamp5(avg_for_start_end?),polyline_id6)
    # to (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),舍弃了object_type,null可视为padding？将维度都统一到8？
    traj_nd = np.hstack([traj_nd, np.zeros((traj_nd.shape[0], 2), dtype=traj_nd.dtype)])
    traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 6]]

    # don't ignore the id
    polyline_features = np.vstack((traj_nd, lane_nd))
    data = [[polyline_features.astype(np.float32), offset_gt,
             traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0]]]
    # traj:19*8,  lane:9*8
    # traj_id2mask保存traj的索引{0:(0,19)}  lane_id2mask保存lane的{1:(0,9),2:(9,18)……}

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TARJ_LEN", "LANE_LEN"]
    )


def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )

if __name__ == "__main__":
    am = ArgoverseMap()
    for folder in os.listdir(DATA_DIR): #循环处理每个数据集文件夹train/test/val
        if not re.search(r'val', folder):
        # FIXME: modify the target folder by hand ('val|train|sample|test')
        # if not re.search(r'test', folder):
            continue
        print(f"folder: {folder}")
        afl = ArgoverseForecastingLoader(os.path.join(DATA_DIR, folder))
        norm_center_dict = {}
        for name in tqdm(afl.seq_list): #对文件夹里的每一个文件分别处理
            afl_ = afl.get(name) #每个csv文件数据
            path, name = os.path.split(name) #文件路径和文件名
            name, ext = os.path.splitext(name) #文件名和后缀名

            agent_feature, obj_feature_ls, lane_feature_ls, norm_center = compute_feature_for_one_seq(
                afl_.seq_df, am, OBS_LEN, LANE_RADIUS, OBJ_RADIUS, viz=False, mode='nearby', folder_name=folder)
            #afl_.seq_df是按行保存的csv文件
            # 处理feature文件
            df = encoding_features(
                agent_feature, obj_feature_ls, lane_feature_ls)
            #每个sence保存一个pkl文件
            save_features(df, name, os.path.join(
                INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))

            norm_center_dict[name] = norm_center

        #保存norm_center文件
        with open(os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}-norm_center_dict.pkl"), 'wb') as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)
            # print(pd.DataFrame(df['POLYLINE_FEATURES'].values[0]).describe())
