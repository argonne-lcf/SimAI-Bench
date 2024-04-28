##### 
##### This script contains general utilities that can be useful
##### to many training applications and driver scripts 
#####

from os.path import exists
import sys
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

### MPI Communicator class
class MPI_COMM:
    def __init__(self):
        """
        MPI Communicator class
        """
        self.comm = None
        self.size = None
        self.rank = None
        self.name = None
        self.rankl = None
        self.sum = None
        self.minloc = None
        self.maxloc = None

    def init(self, cfg, print_hello=False):
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()
        self.rankl = self.rank % (cfg.ppn*cfg.ppd)
        self.sum = MPI.SUM
        self.min = MPI.MIN
        self.max = MPI.MAX
        self.minloc = MPI.MINLOC
        self.maxloc = MPI.MAXLOC
        if print_hello:
            print(f"Hello from MPI rank {self.rank}/{self.size} and local rank {self.rankl}")
            sys.stdout.flush()

    def finalize(self):
        MPI.Finalize()

### Compute the average of a quantity across all ranks
def metric_average(comm, val):
    avg_val = comm.comm.allreduce(val, op=comm.sum)
    avg_val = avg_val / comm.size
    return avg_val


### Compute the correlation coefficient between predicted and target outputs
def comp_corrCoeff(output_tensor, target_tensor):
    target = target_tensor.numpy()
    target = np.ndarray.flatten(target)
    output = output_tensor.detach().numpy()
    output = np.ndarray.flatten(output)
    corrCoeff = np.corrcoef([output,target])
    return corrCoeff


### Count the number of trainable parameters in a model
def count_weights(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params

# Print FOM
def print_fom(time2sol: float, train_data_sz: float, ssim_stats: dict):
    print(f"Time to solution [s]: {time2sol:>.3f}")
    total_sr_time = ssim_stats["tot_meta"]["max"][0] \
                    + ssim_stats["tot_train"]["max"][0]
    rel_sr_time = total_sr_time/time2sol*100
    rel_meta_time = ssim_stats["tot_meta"]["max"][0]/time2sol*100
    rel_train_time = ssim_stats["tot_train"]["max"][0]/time2sol*100
    print(f"Relative total overhead [%]: {rel_sr_time:>.3f}")
    print(f"Relative meta data overhead [%]: {rel_meta_time:>.3f}")
    print(f"Relative train overhead [%]: {rel_train_time:>.3f}")
    string = f": min = {train_data_sz/ssim_stats['train']['max'][0]:>4e} , " + \
             f"max = {train_data_sz/ssim_stats['train']['min'][0]:>4e} , " + \
             f"avg = {train_data_sz/ssim_stats['train']['avg']:>4e}"
    print(f"Train data throughput [GB/s] " + string)

