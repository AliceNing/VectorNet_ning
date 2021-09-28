from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
import utils.config

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


def get_is_track_stationary(track_df: pd.DataFrame) -> bool:
    """Check if the track is stationary.

    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[int(len(vel) / 2)] #取中间时刻的速度
    return True if threshold_vel < VELOCITY_THRESHOLD else False


def get_agent_feature_ls(agent_df, obs_len, norm_center):
    """
    args:
    returns:
        list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
    """
    xys, gt_xys = agent_df[["X", "Y"]].values[:obs_len], agent_df[[
        "X", "Y"]].values[obs_len:]
    xys -= norm_center  # normalize to last observed timestamp point of agent
    gt_xys -= norm_center  # normalize to last observed timestamp point of agent
    xys = np.hstack((xys[:-1], xys[1:]))

    ts = agent_df['TIMESTAMP'].values[:obs_len]
    ts = (ts[:-1] + ts[1:]) / 2

    return [xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0], gt_xys]


def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, has_attr=False, mode='nearby', query_bbox=None):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
        **kwargs: query_bbox= List[int, int, int, int]
    returns:
        list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    '''

    lane_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len - 1]  # 和norm_center值相同
    nearby_lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

    for lane_id in nearby_lane_ids:
        traffic_control = am.lane_has_traffic_control_measure(
            lane_id, city_name)
        is_intersection = am.lane_is_in_intersection(lane_id, city_name)

        centerlane = am.get_lane_segment_centerline(lane_id, city_name)  # 10,3维 包括高度
        # normalize to last observed timestamp point of agent
        centerlane[:, :2] -= norm_center  # 坐标以last_obs为中心

        """得到lane的左右车道线坐标，自定义方法"""
        halluc_lane_1, halluc_lane_2 = get_halluc_lane(centerlane, city_name)
        """得到lane的左右车道线坐标  调用现成的方法-ning"""
        pts = am.get_lane_segment_polygon(lane_id, city_name)
        pts_len = pts.shape[0] // 2
        halluc_lane_1tmp = pts[:pts_len]
        halluc_lane_2tmp = pts[pts_len:2 * pts_len]
        halluc_lane_1tmp[:, :2] -= norm_center
        halluc_lane_2tmp[:, :2] -= norm_center
        # 变成vector形式9*6
        halluc_lane_11tmp = halluc_lane_1tmp[0:9]
        halluc_lane_12tmp = halluc_lane_1tmp[1:]
        halluc_lane_11 = np.hstack((halluc_lane_11tmp, halluc_lane_12tmp))

        halluc_lane_21tmp = halluc_lane_2tmp[0:9]
        halluc_lane_22tmp = halluc_lane_2tmp[1:]
        halluc_lane_21 = np.hstack((halluc_lane_21tmp, halluc_lane_22tmp))

        lane_feature_ls.append(
            [halluc_lane_11, halluc_lane_21, traffic_control, is_intersection, lane_id])

    return lane_feature_ls


def get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, norm_center):
    """
    args:
    returns: list of list, (doubled_track, object_type, timestamp, track_id)
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len - 1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue
        len_remain_df = len(remain_df)
        if len(remain_df) < EXIST_THRESHOLD or get_is_track_stationary(remain_df):  # 如果是静态的或者是长度不够50，就跳过
            continue

        xys, ts = None, None
        # if len(remain_df) < obs_len:
        #     paded_nd = pad_track(remain_df, seq_ts, obs_len, RAW_DATA_FORMAT)
        #     xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
        #     ts = np.array(paded_nd[:, 0], dtype=np.float64)  # FIXME: fix bug: not consider padding time_seq
        # else:
        xys = remain_df[['X', 'Y']].values
        ts = remain_df["TIMESTAMP"].values

        p1 = xys[-1]
        if np.linalg.norm(p0 - p1) > OBJ_RADIUS:  # 筛选obj的范围，超过30就不考虑
            continue

        xys -= norm_center  # normalize to last observed timestamp point of agent
        xys = np.hstack((xys[:-1], xys[1:]))  # 错位，得到vector

        ts = (ts[:-1] + ts[1:]) / 2
        # if not xys.shape[0] == ts.shape[0]:
        #     from pdb import set_trace;set_trace()

        obj_feature_ls.append(
            [xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id])
    return obj_feature_ls