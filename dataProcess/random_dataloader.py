import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class VectorNetDataset(Dataset):

    def __init__(self, data) -> None:
        r"""
        :param data: a list, each element is `[X, Y]` represents the input data and label.
            `X` is a dict, consists `item_num`, `target_id`, `polyline_list`.
                `item_num`: the number of items in this data
                `target_id`: the prediction target index, 0 <= `target_id` < `item_num`
                `polyline_list`: a list, consists `item_num` elements, each element is a set of vectors
                    Note: all data has the same length of vector.
            `Y` is a dict, consists `future`.
                `future`: the future trajectory list
        """
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class RandomDataloader:

    def __init__(self, training_size, test_size, eval_size, v_len, batch_size) -> None:
        self.training_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(training_size, v_len)),
                                                batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(test_size, v_len)),
                                            batch_size=batch_size, shuffle=False)
        self.eval_dataloader = DataLoader(dataset=VectorNetDataset(self.get_random_data(eval_size, v_len)),
                                            batch_size=batch_size, shuffle=False)
 
    def get_random_data(self, N, v_len, item_num_max=10, polyline_size_max=10, future_len=30, data_dim=2):  
        res = []
        for i in range(N):
            ans = dict()
            # item_num = torch.randint(2, item_num_max, (1,))
            item_num =torch.tensor([100], dtype = torch.int32)  #polyline 数量都固定为100
            rot = torch.rand(2, 2)

            polyline_list = []
            for j in range(item_num):
                # v_num = torch.randint(1, polyline_size_max, (1,)).int()
                v_num = 19  #vector 数量都固定为19
                polyline = torch.rand(v_num, v_len)
                polyline_list.append(polyline)

            
            gt_preds = torch.rand(future_len, data_dim)
            has_preds = torch.randint(0, 1, (future_len, 1)).bool()
            ans["item_num"] = item_num
            ans["polyline_list"] = polyline_list
            ans["rot"] = rot
            ans["gt_preds"] = gt_preds
            res.append(ans)
            # print(res)
        return res

if __name__ == '__main__':
    loader = RandomDataloader(2, 0, 0, 9,2)
    for epoch in range(1):
        for i, data in enumerate(loader.training_dataloader):

            print("epoch", epoch, "的第" , i, "个inputs", 
                "item_num:", data["item_num"].data.size(),   #batch_size, 1
                "rot:", data["rot"].data.size(),              #batch_size, 2, 2
                "polyline_list:", len(data["polyline_list"]),  #p_num, batch_size, 19, 9
                "vector_list:", len(data["polyline_list"][0][0]),
                "gt_preds:", data["gt_preds"].data.size())     #batch_size, 30, 2

   # for i in range()