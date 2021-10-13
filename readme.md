## vectorNet

论文地址：https://arxiv.org/pdf/2005.04259.pdf

把车道线和运动轨迹等抽象成不同的节点簇，构成一个subgraph，进行局部的特征融合

再把不同的polyline组成大的graph，卷积后，预测mask掉的节点的feature ，同时预测agent的轨迹，与ground truth进行比较。

### 1.轨迹和地图的表示

​	将一小段时间或者空间内的运动轨迹或车道线，抽象成一个节点，它属于整个轨迹（车道线）折线条图。节点的特征为起点坐标ds=xy，终点坐标de=xy，属性特征a（比如障碍物类型、时间戳、限速等），节点在子图中的ID=j。为了使输入的节点特征不受agent位置的影响，论文将所有向量的坐标归一化，使其以agent目标最后观测时间的位置为中心。

![](images/vectornet_overview.png)

### 2.子图构建

​	子图P中的每个节点都属于同一条折线，P={v1,v2,……,vp}，子图前向传播过程的公式和图解如下：

![](images/vectornet_formu2.png)

![](images/vectornet_fig3.png)

​	stetp1:  genc函数：MLP模型（FC+norm+Relu）。   多层的！！！整个sub-graph也是多层的！！！（3层）

​	step2:  agg函数：邻居节点特征整合，最大池化层。  

​	step3:  rel函数：特征整合，拼接的方式。  

​	整个step1-step3为一层，模型堆叠了多个这样的层。不同层之间的genc参数是不同的，同层内节点之间参数共享。

​	最后整个折线的特征使用agg函数拼接得到。论文中提出当节点的起点和终点相同，并且a和l为空的时候，和PointNet相同。

### 3.全局图构建

​	折线特征更新过程如下，其中Pli(l和i分别为上下标)是第l层的折线特征，A是不同折线之间交互的全局图，这里使用了全连接的图。

![](images/vectornet_formu4.png)

![](images/vectornet_formu5.png)

​	GNN是attention的计算方式，先分别经过线性变换得到PQ，PK，PV，再PQ和PK相乘经过softmax后得到注意力系数，最后累加求和，更新特征。推理时，只需要更新agent的节点特征。

​	在经过t层GNN之后(代码里使用了1层，可以堆叠多层)，得到的折线特征再使用一个MLP进行解码，来预测未来的运动轨迹。

​	在训练时，模型也随机Mask掉了一些Polyline_node特征，使用一个MLP解码器进行预测，以提升模型性能。这个部分不在推理时使用。为了区分mask掉的node，会根据其最小的起点算一个标识特征Pid，故初始node特征为：

![](images/vectornet_formu8.png)

### 4.损失函数

​	损失函数包括两部分，预测轨迹的negative Gaussian log-likelihood损失和node特征重构的Huber损失。node 特征在进入Global net之前做了L2归一化。

预测的轨迹和LaneGCN相同，都是相对最后时刻中心位置的相对位移，然后进行坐标系转换和ground truth值进行比较。

## 代码实现

### 1.数据预处理部分  

代码结构参考lanegcn的预处理过程  

**处理过程**：  

​	时间戳归一化（减去最小值）  

​	位置归一化标准（根据last_obs的位置）并rot  

​	traj的特征：xs, ys, xe, ye, type, start_time, end_time, speed, polyline_id（不足20的补0）  

​	lane的特征：xs, ys, xe, ye, type, traffic_control, turn_direction, intersection, polyline_id  

其中：type：AGENT=0， ctx_traj=1，lane=2  

​			direction:  left=1，right=2，None=0  

​			 traffic_control,intersection：true=1， false=0  

**最终data保存**：  

​			'item_num'：polyline数量  

​			'polyline_list'：polyline的特征，每个polyline包含的vector数量不同，但vector的特征长度都为9  
​			'rot'：保存旋转矩阵  
​			'gt_preds'：保存agent的future 坐标   
​			'has_preds'：未来30个时刻gt_preds是否有值  
​			'idx'：data index  
### 2.模型框架
代码参考 https://github.com/DQSSSSS/VectorNet
BUG：sub-graph 为三层，每一层里的genc函数也是堆叠了三层的MLP
###To do list
- [ ] 模型改进(加tensorboard)
- [ ] 模型改进(并行)
- [ ] 模型改进(结果可视化)
