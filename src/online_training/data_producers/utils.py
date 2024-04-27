from typing import Tuple, Optional
from gmpy2 import is_square
import numpy as np
import math

PI = math.pi

# Generate training data for each model
def generate_training_data(args, comm_info: Tuple[int,int], 
                           step: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate training data for each model
    """
    rank = comm_info[0]
    size = comm_info[1]
    random_seed = 12345 + 1000*rank
    rng = np.random.default_rng(seed=random_seed)
    if (args.problem_size=="debug"):
        n_samples = 512
        ndIn = 1
        ndTot = 2
        coords = rng.uniform(low=0.0, high=2*PI, size=n_samples)
        y = np.sin(coords)+0.1*np.sin(4*PI*coords)
        y = (y - (-1.0875)) / (1.0986 - (-1.0875)) # min-max scaling
        data = np.vstack((coords,y)).T
    elif (args.problem_size=="small"):
        assert is_square(size) or size==1, "Number of MPI ranks must be square or 1"
        N = 32
        n_samples = N**2
        ndIn = 1
        ndTot = 2
        x, y = partition_domain((-2*PI, 2*PI), (-2*PI, 2*PI), N, size, rank)
        x, y = np.meshgrid(x, y)
        coords = np.vstack((x.flatten(),y.flatten())).T
        u = np.sin(0.1*step)*np.sin(x)*np.sin(y)
        udt = np.sin(0.1*(step+1))*np.sin(x)*np.sin(y)
        data = np.vstack((u.flatten(),udt.flatten())).T

    return_dict = {
        "n_samples": n_samples,
        "n_dim_in": ndIn,
        "n_dim_tot": ndTot
    }
    return data, coords, return_dict

# Partition the global domain
def partition_domain(x_lim: Tuple[float,float], y_lim: Tuple[float,float],
                     N: int, comm_size: int, 
                     rank: int) -> Tuple[np.ndarray,np.ndarray]:
    if (comm_size==1):
        x = np.linspace(x_lim[0], x_lim[1], N)
        y = np.linspace(y_lim[0], y_lim[1], N)
    else:
        n_parts_per_dim = math.isqrt(comm_size)
        xrange = (x_lim[1]-x_lim[0])/n_parts_per_dim
        x_id = rank % n_parts_per_dim
        x_min = xrange*x_id
        x_max = xrange*(x_id+1)
        x = np.linspace(x_min, x_max, N)
        yrange = (y_lim[1]-y_lim[0])/n_parts_per_dim
        y_id = rank // n_parts_per_dim
        y_min = yrange*y_id
        y_max = yrange*(y_id+1)
        y = np.linspace(y_min, y_max, N)
    return x, y

# Print FOM
def print_fom(time2sol: float, train_data_sz: float, ssim_stats: dict):
    print(f"Time to solution [s]: {time2sol:>.3f}")
    total_sr_time = ssim_stats["tot_meta"]["max"][0] \
                    + ssim_stats["tot_train"]["max"][0] \
                    + ssim_stats["tot_infer"]["max"][0]
    rel_sr_time = total_sr_time/time2sol*100
    rel_meta_time = ssim_stats["tot_meta"]["max"][0]/time2sol*100
    rel_train_time = ssim_stats["tot_train"]["max"][0]/time2sol*100
    rel_infer_time = ssim_stats["tot_infer"]["max"][0]/time2sol*100
    print(f"Relative total overhead [%]: {rel_sr_time:>.3f}")
    print(f"Relative meta data overhead [%]: {rel_meta_time:>.3f}")
    print(f"Relative train overhead [%]: {rel_train_time:>.3f}")
    print(f"Relative infer overhead [%]: {rel_infer_time:>.3f}")
    string = f": min = {train_data_sz/ssim_stats['train']['max'][0]:>4e} , " + \
             f"max = {train_data_sz/ssim_stats['train']['min'][0]:>4e} , " + \
             f"avg = {train_data_sz/ssim_stats['train']['avg']:>4e}"
    print(f"Train data throughput [GB/s] " + string)

