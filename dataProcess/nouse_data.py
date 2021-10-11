from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
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
config["LANE_RADIUS"] = 30 #30 [100, -100, 100, -100]
config["OBJ_RADIUS"] = 30
config["DATA_DIR"] = './data'
config["OBS_LEN"] = 20
config["Preprocess_DATA_DIR"] = './preprosessed_data'
config["query_bbox"] = [-100, 100, -100, 100]
config["viz"] = False

def compute_velocity(track_df: pd.DataFrame) -> List[float]:
    """Compute velocities for the given track.

    Args:
        track_df (pandas Dataframe): Data for the track
    Returns:
        vel (list of float): Velocity at each timestep

    """
    x_coord = track_df["X"].values
    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values
    # tmp_vel = [(   #zip(*)将行的列表转换为列的列表
    #     x_coord[i] - x_coord[i - 1] /
    #     (float(timestamp[i]) - float(timestamp[i - 1])),
    #     y_coord[i] - y_coord[i - 1] /
    #     (float(timestamp[i]) - float(timestamp[i - 1])),
    # ) for i in range(1, len(timestamp))]
    # vel_x_0, vel_y_0 = zip(*tmp_vel)
    vel_x, vel_y = zip(*[(   #zip(*)将行的列表转换为列的列表,最开始是[(x,y),(x,y)……(x,y)]的形式，然后改为[(x,x,x……x)(y,y……y)]的形式
        x_coord[i] - x_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        y_coord[i] - y_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]

    return vel


def get_is_track_stationary(track_df: pd.DataFrame, config) -> bool:
    """Check if the track is stationary.

    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[int(len(vel) / 2)] #取中间时刻的速度
    return True if threshold_vel < config["VELOCITY_THRESHOLD"] else False


def get_agent_feature_ls(agent_df, obs_len, norm_center, rot):
    """
    args:
    returns:
        list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        return:list，其包含xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0], gt_xys
    其中xys是前20个时刻的vector（减去norm_center了），ts是vector两点的平均 时间戳，gt_xys是后30个时刻的位置（减去norm_center了）
    """
    xys, gt_xys = agent_df[["X", "Y"]].values[:obs_len], agent_df[[
        "X", "Y"]].values[obs_len:]
    # xys -= norm_center  # normalize to last observed timestamp point of agent
    xys = np.matmul(rot, (xys - norm_center).T).T
    # gt_xys -= norm_center  # normalize to last observed timestamp point of agent
    xys = np.hstack((xys[:-1], xys[1:]))

    ts = agent_df['TIMESTAMP'].values[:obs_len]
    ts = (ts[:-1] + ts[1:]) / 2

    return [xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0], gt_xys]


def get_nearby_lane_feature_ls(am, agent_df, config, city_name, norm_center, rot, has_attr=False):
    '''
    根据最后一个观察点得到上下文的Lane特征

    返回值:lane特征列表
    list长度是lane_num,每个里面包含一个长度为5的list:
    格式：[left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    # 其中左右车道线坐标是三维的，包括高度信息（前两维减去norm_center,并旋转）

    '''

    lane_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[config["OBS_LEN"] - 1]  # 和norm_center值相同
    nearby_lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, config["LANE_RADIUS"])

    for lane_id in nearby_lane_ids:
        traffic_control = am.lane_has_traffic_control_measure(
            lane_id, city_name)
        is_intersection = am.lane_is_in_intersection(lane_id, city_name)

        centerlane = am.get_lane_segment_centerline(lane_id, city_name)  # 10,3维 包括高度
        # normalize to last observed timestamp point of agent and rot
        centerlane[:, :2] = np.matmul(rot, (centerlane[:, :2] - norm_center).T).T
        # centerlane[:, :2] -= norm_center  # 坐标以last_obs为中心

        """得到lane的左右车道线坐标  调用现成的方法-ning"""
        pts = am.get_lane_segment_polygon(lane_id, city_name)
        pts_len = (pts.shape[0]-1) // 2
        if pts_len != 10:
            print(pts_len)
        lane_1tmp = pts[:pts_len]
        lane_2tmp = pts[pts_len:2 * pts_len]
        # halluc_lane_1tmp[:, :2] -= norm_center
        # halluc_lane_2tmp[:, :2] -= norm_center
        lane_1tmp[:, :2] = np.matmul(rot, (lane_1tmp[:, :2] - norm_center).T).T
        lane_2tmp[:, :2] = np.matmul(rot, (lane_2tmp[:, :2] - norm_center).T).T

        # 变成vector形式9*6
        lane_1tmp_1 = lane_1tmp[0:(pts_len-1)]
        lane_1tmp_2 = lane_1tmp[1:]
        # print(lane_1tmp_1.shape)
        # print(lane_1tmp_2.shape)
        lane_1 = np.hstack((lane_1tmp_1, lane_1tmp_2))

        lane_2tmp_1 = lane_2tmp[0:(pts_len-1)]
        lane_2tmp_2 = lane_2tmp[1:]
        lane_2 = np.hstack((lane_2tmp_1, lane_2tmp_2))

        lane_feature_ls.append(
            [lane_1, lane_2, traffic_control, is_intersection, lane_id])

    return lane_feature_ls


def get_nearby_moving_obj_feature_ls(agent_df, traj_df, config, seq_ts, norm_center, rot):
    """
    args:
    returns: list of list, (doubled_track, object_type, timestamp, track_id)
    return: xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id
    其中xys是vector（减去norm_center了）,obj类型，ts是vector两点的平均时间戳，track_id
    xys和ts的长度都是49！！
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[config["OBS_LEN"] - 1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue
        len_remain_df = len(remain_df)
        # 如果是静态的或者是长度不够50，就跳过
        if len(remain_df) < config["EXIST_THRESHOLD"] or get_is_track_stationary(remain_df, config):
            continue

        xys, ts = None, None
        xys = remain_df[['X', 'Y']].values
        ts = remain_df["TIMESTAMP"].values

        p1 = xys[-1]
        if np.linalg.norm(p0 - p1) > config["OBJ_RADIUS"]:  # 筛选obj的范围，超过30就不考虑
            continue

        # xys -= norm_center  # normalize to last observed timestamp point of agent
        xys = np.matmul(rot, (xys - norm_center).T).T
        xys = np.hstack((xys[:-1], xys[1:]))  # 错位，得到vector

        ts = (ts[:-1] + ts[1:]) / 2  #时刻取中
        obj_feature_ls.append(
            [xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id])
    return obj_feature_ls

def compute_feature_for_one_seq(traj_df, am, config, viz = False, folder_name = "") -> List[List]:
    """
    return lane & track features
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center np.ndarray: (2, )
    """
    # 时间戳归一化
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)
    seq_len = seq_ts.shape[0]
    city_name = traj_df['CITY_NAME'].iloc[0]

    agent_df = None
    end_x, end_y, start_x, start_y, query_x, query_y, norm_center = [None] * 7  # 起点，终点和中间20个时刻的观察点

    # 得到agent的轨迹，旋转矩阵，起始点等信息
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        if obj_type == 'AGENT':  # AGENT对象的
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            end_x, end_y = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[config["OBS_LEN"] - 1]
            pre_x, pre_y = agent_df[['X', 'Y']].values[config["OBS_LEN"] - 2]
            norm_center = np.array([query_x, query_y])
            pre_center = np.array([pre_x, pre_y])
            # 计算旋转角和旋转矩阵
            pre = pre_center - norm_center
            theta = np.pi - np.arctan2(pre[1], pre[0])
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # 测试集把后30个时间戳的数据删掉，只留前20个时刻的数据
    if folder_name == "test":
        traj_df = traj_df[traj_df['TIMESTAMP'] <= agent_df['TIMESTAMP'].values[obs_len - 1]]

        assert (np.unique(traj_df["TIMESTAMP"].values).shape[0] == obs_len), "Obs len mismatch"

    lane_feature_ls = get_nearby_lane_feature_ls(am, agent_df, config, city_name, norm_center, rot)

    obj_feature_ls = get_nearby_moving_obj_feature_ls(agent_df, traj_df, config, seq_ts, norm_center, rot)

    agent_feature = get_agent_feature_ls(agent_df, config["OBS_LEN"], norm_center, rot)

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

        plt.plot(end_x - query_x, end_y - query_y, 'o',
                 color=config["color_dict"]['AGENT'], markersize=7)
        plt.plot(0, 0, 'x', color='blue', markersize=4)
        plt.plot(start_x - query_x, start_y - query_y,
                 'x', color='blue', markersize=4)
        plt.show()

    return [agent_feature, obj_feature_ls, lane_feature_ls, norm_center, rot]

def trans_gt_offset_format(gt):
    """#变成每一步移动的距离
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets,
    starting from the last observed location.We rotate the coordinate system
    based on the heading of the target vehicle at the last observed location.

    """
    assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1])) #变成每一步移动的距离

    assert (offset_gt.cumsum(axis=0) - gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) - gt).sum()}"

    return offset_gt

def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls, config):
    """
    args:
        agent_feature_ls:
            (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
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

    '''agent feature'''
    pre_traj_len = traj_nd.shape[0]
    agent_len = agent_feature[0].shape[0]  # agent obs vector len
    agent_nd = np.hstack((agent_feature[0], np.ones(
        (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    # agent_obseve_seq4,one1,time1,polyline_id1=7维
    assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((traj_nd, agent_nd))
    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1

    '''obj feature'''
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

    '''lane feature'''
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

    '''handling `nan` in lane_nd'''
    col_mean = np.nanmean(lane_nd, axis=0)
    if np.isnan(col_mean).any():
        # raise ValueError(
        # print(f"{col_mean}\nall z (height) coordinates are `nan`!!!!")
        lane_nd[:, 2].fill(.0)
        lane_nd[:, 5].fill(.0)
    else:
        inds = np.where(np.isnan(lane_nd))
        lane_nd[inds] = np.take(col_mean, inds[1])

    # offset_gt = trans_gt_offset_format(gt)  #gt变成每一步移动的距离
    offset_gt = gt  #gt使用真实没有任何处理的坐标

    '''change lanes feature '''
    # from (xs, ys, zs, xe, ye, ze, polyline_id) to (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
    lane_nd = np.hstack([lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])  # timestap???
    lane_nd = lane_nd[:, [0, 1, 3, 4, 7, 2, 5, 6]]  # x1,y1,x2,y2,0,z1,z2,polyline_id

    '''change object features'''
    # from (xs0, ys1, xe2, ye3, obejct_type4, timestamp5(avg_for_start_end?),polyline_id6)
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
    dir = os.listdir(config["DATA_DIR"])
    for folder in os.listdir(config["DATA_DIR"]): #循环处理每个数据集文件夹train/test/val
        print(f"folder: {folder}")
        afl = ArgoverseForecastingLoader(os.path.join(config["DATA_DIR"], folder))
        norm_center_dict = {}
        rot_dict = {}
        for name in tqdm(afl.seq_list): #对文件夹里的每一个文件分别处理
            # print(name)
            afl_ = afl.get(name) #每个csv文件数据
            path, name = os.path.split(name) #文件路径和文件名
            name, ext = os.path.splitext(name) #文件名和后缀名

            agent_feature, obj_feature_ls, lane_feature_ls, norm_center, rot = compute_feature_for_one_seq(
                afl_.seq_df, am, config, viz=False, folder_name = folder)
            #afl_.seq_df是按行保存的csv文件

            # 处理feature文件
            df = encoding_features(agent_feature, obj_feature_ls, lane_feature_ls, config)

            #每个sence保存一个pkl文件
            save_features(df, name, os.path.join(config["Preprocess_DATA_DIR"], f"{folder}_intermediate"))

            norm_center_dict[name] = norm_center
            rot_dict[name] = rot


        #保存norm_center文件 rot文件
        with open(os.path.join(config["Preprocess_DATA_DIR"], f"{folder}-norm_center_dict.pkl"), 'wb') as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)
            # print(pd.DataFrame(df['POLYLINE_FEATURES'].values[0]).describe())
        with open(os.path.join(config["Preprocess_DATA_DIR"], f"{folder}-rot_dict.pkl"), 'wb') as f:
            pickle.dump(rot_dict, f, pickle.HIGHEST_PROTOCOL)