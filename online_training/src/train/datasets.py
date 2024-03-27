##### 
##### This script contains all the Pytorch Datasets that might be needed for
##### different training approaches and algorithms 
#####

import torch
from torch.utils.data import Dataset

### Dataset that returns as length the number of tensors in the database sent by
### PHASTA, and as item the key to a particular tensor, where the key is tagged
### with both the rank and time step number
### It takes as input the number of total tensors in all databases (in case of colodated DB), 
### the time step number and the head rank of PHASTA on each node
### Used to pull a particular tensor from the database
class PhastaRankDataset_StepNum(Dataset):
    def __init__(self, num_tot_tensors, step_num, head_rank):
        self.total_data = num_tot_tensors
        self.step = step_num
        self.head_rank = head_rank

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        return f"y.{tensor_num}.{self.step}"


### Dataset that returns as length the number of tensors in the database sent by
### PHASTA, and as item the key to a particular tensor, where the key is tagged
### with either only the rank number or both the tag and step number
### It takes as input the number of total tensors in all databases (in case of colodated DB), 
### the time step number, the head rank of PHASTA on each node,
### and a flag determining whether tensors are being overwritten in database
### Used to pull a particular tensor from the database
class PhastaRankDataset(Dataset):
    def __init__(self, num_tot_tensors, step_num, head_rank, dataOverWrite):
        self.total_data = num_tot_tensors
        self.step = step_num
        self.head_rank = head_rank
        self.ow = dataOverWrite

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        if (self.ow > 0.5):
            return f"y.{tensor_num}"
        else:
            return f"y.{tensor_num}.{self.step}"


### Dataset that returns as length the length of the tensor passed in (number of rows)
### and as index the index of that tensor.
### The input is a concatenated tensor of training data (concatenated after pulling entire
### tensors from the database).
### Used to split training data loaded onto ML ranks into mini-batches
class MiniBatchDataset(Dataset):
    def __init__(self,concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]



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
        

### Dataset that takes in a list of rank numbers and returns the length of the list
### and the key for an index in that list of ranks
### The input is a list of rank numbers corresponding to tensors in the database
### Used when the ranks are pre-determined and passed in as a list, was made
### to enable easy splitting of database tensors into training, validation, testing
class KeyMFDataset(Dataset):
    def __init__(self,tensor_list,nrankl,head_rank,filters):
        self.tensors = tensor_list
        self.size = nrankl
        self.head_rank = head_rank
        self.filters = filters

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        nfilters = self.filters.size
        rank_id = idx//size
        filt_id = idx%nfilters
        delta = round(filters[filt_id])
        return f"y.{rank_id}.{delta}"
    

### Dataset for offline training
### Classic map-stype dataset that returns the indexed array
class OfflineDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]



