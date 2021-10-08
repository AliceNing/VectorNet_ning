from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os

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
        pts_len = pts.shape[0] // 2
        lane_1tmp = pts[:pts_len]
        lane_2tmp = pts[pts_len:2 * pts_len]
        # halluc_lane_1tmp[:, :2] -= norm_center
        # halluc_lane_2tmp[:, :2] -= norm_center
        lane_1tmp[:, :2] = np.matmul(rot, (lane_1tmp[:, :2] - norm_center).T).T
        lane_2tmp[:, :2] = np.matmul(rot, (lane_2tmp[:, :2] - norm_center).T).T

        # 变成vector形式9*6
        lane_1tmp_1 = lane_1tmp[0:9]
        lane_1tmp_2 = lane_1tmp[1:]
        lane_1 = np.hstack((lane_1tmp_1, lane_1tmp_2))

        lane_2tmp_1 = lane_2tmp[0:9]
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