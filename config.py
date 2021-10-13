import os
import torch
root_path = os.path.dirname(os.path.abspath(__file__))

config = dict()
config["query_bbox"] = [-100, 100, -100, 100]
config["batch_size"] = 2
config["workers"] = 0
config["val_workers"] = 0#32
config["train_dir"] = "../data/train"   #os.path.join(root_path, "data/train")
config["val_dir"] = "./data/val"       #os.path.join(root_path, "./data/val")
config["test_dir"] = "./data/test"     #os.path.join(root_path, "./data/test")
# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "dataset", "preprocess", "train.p")
config["preprocess_val"] = os.path.join(root_path, "dataset", "preprocess", "val.p")
config['preprocess_test'] = os.path.join(root_path, "dataset", 'preprocess', 'test.p')
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
config['device'] = device