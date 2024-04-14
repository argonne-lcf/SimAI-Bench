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
    def __init__(self, num_db_tensors, head_rank):
        self.total_data = num_db_tensors
        self.head_rank = head_rank

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        return f"x.{tensor_num}"
    
### Map-style dataset that returns the tensor key string to grab from DB
# Generates a key for DB tensor with varying rank ID and time step number
class RankStepDataset(torch.utils.data.Dataset):
    def __init__(self, num_db_tensors, steps, head_rank):
        self.ranks = num_db_tensors
        self.steps = steps
        self.num_steps = len(steps)
        self.head_rank = head_rank
        self.total_data = self.ranks*self.num_steps

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        rank_id = idx%self.ranks
        rank_id = rank_id+self.head_rank
        step_id = idx//self.ranks
        step = self.steps[step_id]
        return f"x.{rank_id}.{step}"

"""
### Dataset that takes in a list of rank numbers and returns the length of the list
### and the key for an index in that list of ranks
### The input is a list of rank numbers corresponding to tensors in the database
### Used when the ranks are pre-determined and passed in as a list, was made
### to enable easy splitting of database tensors into training, validation, testing
class KeyDataset(Dataset):
    def __init__(self,rank_list,head_rank,step_num,dataOverWrite):
        self.ranks = rank_list
        self.head_rank = head_rank
        self.step = step_num
        self.ow = dataOverWrite

    def __len__(self):
        return len(self.ranks)

    def __getitem__(self, idx):
        tensor_num = self.ranks[idx]+self.head_rank
        if (self.ow > 0.5):
            return f"y.{tensor_num}"
        else:
            return f"y.{tensor_num}.{self.step}"
"""

