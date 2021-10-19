import os
import torch
root_path = os.path.dirname(os.path.abspath(__file__))

config = dict()
config["query_bbox"] = [-100, 100, -100, 100]
config["preprocess_batch_size"] = 1
config["batch_size"] = 2
config["workers"] = 0
config["val_workers"] = 0#32

#  data save path
config["train_dir"] = os.path.join(root_path, "data", "train")   #os.path.join(root_path, "data/train")
config["val_dir"] = os.path.join(root_path, "data", "val")       #os.path.join(root_path, "./data/val")
config["test_dir"] = os.path.join(root_path, "data", "test")     #os.path.join(root_path, "./data/test")
config["save_dir"] = os.path.join(root_path, "data", "save_model")    #os.path.join(root_path, "./data/test")

#   Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "data", "preprocess", "train.p")
config["preprocess_val"] = os.path.join(root_path, "data", "preprocess", "val.p")
config['preprocess_test'] = os.path.join(root_path, "data", 'preprocess', 'test.p')

#   cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
config['device'] = device

# print(torch.device)