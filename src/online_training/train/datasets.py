##### 
##### This script contains all the Pytorch Datasets that might be needed for
##### different training approaches and algorithms 
#####

import torch
from torch.utils.data import Dataset

### Classic map-stype dataset that returns the indexed array
class MiniBatchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

### Map-style dataset that returns the tensor key string to grab from DB
# Generates a key for DB tensor with varying rank ID only
class RankDataset(torch.utils.data.Dataset):
    def __init__(self, num_db_tensors, head_rank, model, rank):
        self.model = model
        self.rank = rank
        self.head_rank = head_rank
        self.total_data = num_db_tensors

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        if self.model=='mlp':
            tensor_num = idx+self.head_rank
        elif self.model=='gnn':
            tensor_num = self.rank
        return f"x.{tensor_num}"
    
### Map-style dataset that returns the tensor key string to grab from DB
# Generates a key for DB tensor with varying rank ID and time step number
class RankStepDataset(torch.utils.data.Dataset):
    def __init__(self, num_db_tensors, steps, head_rank, model, rank):
        self.ranks = num_db_tensors
        self.model = model
        self.rank = rank
        self.head_rank = head_rank
        self.steps = steps
        self.num_steps = len(steps)
        self.total_data = self.ranks*self.num_steps

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        if self.model=='mlp':
            rank_id = idx%self.ranks
            rank_id = rank_id+self.head_rank
            step_id = idx//self.ranks
        elif self.model=='gnn':
            rank_id = self.rank
            step_id = idx
        step = self.steps[step_id]
        return f"x.{rank_id}.{step}"



