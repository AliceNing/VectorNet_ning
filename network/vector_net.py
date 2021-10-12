import copy
import torch
from torch import nn
from config import *
from util import *
from network.mlp import MLP
from network.global_graph import GlobalGraph
from network.sub_graph import SubGraph
import torch.nn.functional as F

class VectorNetWithPredicting(nn.Module):
    r"""
    A class for packaging the VectorNet and future trajectory prediction module.
    The future trajectory prediction module uses MLP.
    """

    def __init__(self, v_len, time_stamp_number, data_dim=2):
        r"""
        Construct a VectorNet with predicting.
        
        Args:
            v_len: length of vector
            time_stamp_number: the length of time stamp for predicting the future trajectory.
            data_dim: dimension of dataset, effects the output coordinate
        """
        super(VectorNetWithPredicting, self).__init__()
        self.vector_net = VectorNet(v_len)
        self.traj_decoder = MLP(input_size=self.vector_net.p_len,
                                    output_size=time_stamp_number * data_dim)
        self.dim = data_dim

    def forward(self, data):
        r"""
        Args:
            item_num: [batch_size, 1], number of items
            target_id: [batch_size, 1], prediction agent id
            polyline_list: list of polyline
        Returns:
            future trajectory vector of `target_id`,
            shape is [batch_size, time_stamp_number, data_dim], pre-step offset
        """
        data = gpu(data)
        feature = self.vector_net(data['item_num'], data['polyline_list'])
        x = self.traj_decoder(feature)
        x = x.view(x.shape[0], -1, self.dim) # [batch_size, time_stamp_number, data_dim]
        return x


class VectorNet(nn.Module):
    r""" 
    Vector network.
    """

    def __init__(self, v_len, sub_layers=3, global_layers=1):
        r"""
        Construct a VectorNet.
        Args:
            v_len (int): length of each vector v ([ds,de,a,j]).
            sub_layers (int): 
            global_layers (int): 
        """
        super(VectorNet, self).__init__()
        self.sub_graph = SubGraph(layers_number=sub_layers, v_len=v_len)
        self.p_len = v_len * (2 ** sub_layers)
        self.global_graph = GlobalGraph(len=self.p_len, layers_number=global_layers)

    def forward(self, item_num, polyline_list):#  item_num: batch_size list,  polyline_list: batch_size list
        r"""
        Note: Because different data has different number of agents, different agent has different number of vectors, so we
        choose batch_size=1
        
        Args:
            item_num (Tensor): [batch_size, 1], number of items
            target_id (Tensor, dtype=torch.int64): [batch_size, 1], prediction agent id
            polyline_list (list): list of polyline
        
        Returns: 
            A tensor represents the embedding of prediction agent,
            shape is [batch_size, self.p_len]
        """
        # batch_size = item_num.shape[0]
        batch_size = len(item_num)

        p_batch_list = []
        for i in range(batch_size):
            p_list = []
            for polyline in polyline_list[i]:
                # polyline = torch.cat([polyline] * batch_size, axis=0)  # [batch_size, v_number, v_len]
                polyline = torch.unsqueeze(polyline, 0)  #1,19,9
                p = self.sub_graph(polyline)  # [batch_size, p_len]  1, 72
                p_list.append(p)
            p_list = torch.stack(p_list, axis=1) # [batch_size, p_number, p_len]
            # p_list = torch.squeeze(p_list, 1)
            p_batch_list.append(p_list)


        # P = torch.stack(p_list, axis=1) # [batch_size, p_number, p_len]
        # assert P.shape == (batch_size, len(polyline_list), self.p_len)

        feature_batch = []
        for i in range(batch_size):
            feature = self.global_graph(p_batch_list[i])  # [batch_size, p_len]
            feature_batch.append(feature)
        # assert feature.shape == (batch_size, self.p_len)
        feature_batch = torch.stack(feature_batch, 0)
        feature_batch = torch.squeeze(feature_batch, 1)
        return feature_batch

