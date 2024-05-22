import sys
import os, os.path
import io
from typing import Optional
from time import perf_counter
import logging
import numpy as np
import torch
try:
    from torch_geometric.nn import knn_graph
except:
    pass

import dragon
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict

# Dragon Client Class for the Simulation (Data Producer)
class Dragon_Sim_Client:
    def __init__(self, args, rank: int, size: int):
        self._dd_serialized = args.dictionary
        self.launch = args.launch
        self.rank = rank
        self.size = size
        self.rankl = self.rank%self.size
        self.ppn = args.ppn
        self.ow = True if args.problem_size=="debug" else False
        self.max_mem = args.db_max_mem_size*1024*1024*1024
        self.model = args.model
        self._dd = None

        self.times = {
            "init": 0.,
            "tot_meta": 0.,
            "tot_train": 0.,
            "train": [],
            "tot_infer": 0.,
            "infer": []
        }
        self.time_stats = {}

        if (self.launch == "colocated"):
            self.head_rank = self.rank//self.ppn
        elif (self.launch == "clustered"):
            self.ppn = size
            self.head_rank = 0

        # For GNN inference
        self.coords = None
        self.edge_index = None

    # Initialize client
    def init(self):
        tic = perf_counter()
        self._dd = DDict.attach(self._dd_serialized, timeout=30)
        toc = perf_counter()
        self.times["init"] = toc - tic

    # Destroy the client
    def destroy(self):
        """Destroy the client
        """
        self._dd.detach()

    # General method to put an object to the Dictionary
    def put(self, key: str, val) -> None:
        self._dd[key] = val

    # General method to get an object from the Dictionary
    def get(self, key: str):
        return self._dd[key]
    
    # General method to see if key exists in the Dictionary
    def key_exists(self, key: str) -> bool:
        return key in self._dd.keys()

    # Set up training case and write metadata
    def setup_training_problem(self, coords: np.ndarray, 
                               data_info: dict) -> None:
        if self.rank==self.head_rank:
            # Run-check
            arr = np.array([1], dtype=np.int64)
            tic = perf_counter()
            self.put('sim_run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

            # Training data setup
            dataSizeInfo = np.empty((6,), dtype=np.int64)
            dataSizeInfo[0] = data_info["n_samples"]
            dataSizeInfo[1] = data_info["n_dim_tot"]
            dataSizeInfo[2] = data_info["n_dim_in"]
            dataSizeInfo[3] = self.size
            dataSizeInfo[4] = self.ppn
            dataSizeInfo[5] = self.head_rank
            tic = perf_counter()
            self.put('sizeInfo', dataSizeInfo)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

            # Write overwrite tensor
            if self.ow:
                arr = np.array([1], dtype=np.int64)
            else:
                arr = np.array([0], dtype=np.int64)
            tic = perf_counter()
            self.put('tensor-ow', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

        # Set up graph if training GNN
        if self.model=='gnn':
            self.setup_graph(coords)

    # Set up training case and write metadata
    def setup_graph(self, coords: np.ndarray):
        if coords.ndim<2:
            coords = np.expand_dims(coords, axis=1)
        self.coords = coords
        self.edge_index = knn_graph(torch.from_numpy(coords), k=2, loop=False).numpy()
        tic = perf_counter()
        self.put(f'pos_node_{self.rank}', self.coords)
        self.put(f'edge_index_{self.rank}', self.edge_index)
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
    
    # Signal to training sim is exiting
    def stop_train(self):
        if self.rank==self.head_rank:
            arr = np.array([0], dtype=np.int64)
            tic = perf_counter()
            self.put('sim-run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
        
    # Send training snapshot
    def send_snapshot(self, array: np.ndarray, step: int):
        if self.ow:
            key = 'x.'+str(self.rank)
        else:
            key = 'x.'+str(self.rank)+'.'+str(step)
        if (self.rank==0):
            print(f'\tSending training data with key {key} and shape {array.shape}', flush=True)
        tic = perf_counter()
        self.put(key, array)
        toc = perf_counter()
        self.times["tot_train"] += toc - tic
        self.times["train"].append(toc - tic)

    # Check DB memory
    def check_mem(self, array: np.ndarray) -> bool:
        key = 'asdf_{self.rank}'
        try:
            tic = perf_counter()
            self.put(key,array)
            #del self._dd[key]
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
            return True
        except:
            return False

    # Send time step
    def send_step(self, step: int):
        if self.rank==self.head_rank:
            step_arr = np.array([step], dtype=np.int64)
            tic = perf_counter()
            self.put('step', step_arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
    
    # Check to see if model exists in DB
    def model_exists(self, comm) -> bool:
        tic = perf_counter()
        local_exists = 1 if self.key_exists(self.model) else 0
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
        global_exists = comm.allreduce(local_exists)
        if global_exists==self.size:
            return True
        else:
            return False
        
    # Perform inference with model on DB
    def infer_model(self, comm, inputs: np.ndarray,
                    outputs: np.ndarray) -> float:
        if inputs.ndim<2:
            inputs = np.expand_dims(inputs, axis=1)
        tic = perf_counter()
        model_bytes = self.get(self.model)
        buffer = io.BytesIO(model_bytes)
        model_jit = torch.jit.load(buffer, map_location='cpu')
        x = torch.from_numpy(inputs).type(torch.float32)
        if (self.model=="gnn"):
            edge_index = torch.from_numpy(self.edge_index).type(torch.int64)
            pos = torch.from_numpy(self.coords).type(torch.float32)
            with torch.no_grad():
                model_jit.to(f'cuda:{3-self.rankl}')
                x = x.to(f'cuda:{3-self.rankl}')
                edge_index = edge_index.to(f'cuda:{3-self.rankl}')
                pos = pos.to(f'cuda:{3-self.rankl}')
                pred = model_jit(x, edge_index, pos).cpu().numpy()
        elif (self.model=='mlp'):
            with torch.no_grad():
                model_jit.to(f'cuda:{3-self.rankl}')
                x = x.to(f'cuda:{3-self.rankl}')
                pred = model_jit(x).cpu().numpy()
        toc = perf_counter()
        self.times["tot_infer"] += toc - tic
        self.times["infer"].append(toc - tic)
        local_mse = ((outputs.flatten() - pred.flatten())**2).mean()
        avg_mse = comm.allreduce(local_mse)/self.size
        return avg_mse
    
    # Check status of training
    def check_train_status(self) -> None:
        while True:
            if self.key_exists('train-run'):
                break

    # Collect timing statistics across ranks
    def collect_stats(self, comm, mpi_ops):
        for _, (key, val) in enumerate(self.times.items()):
            if type(val)==list:
                if val:
                    if 'infer' in key: val.pop(0)
                    collected_arr = np.zeros((len(val)*comm.Get_size()))
                    comm.Gather(np.array(val),collected_arr,root=0)
                    avg = np.mean(collected_arr)
                    std = np.std(collected_arr)
                    minn = np.amin(collected_arr); min_loc = [minn, 0]
                    maxx = np.amax(collected_arr); max_loc = [maxx, 0]
                    summ = np.sum(collected_arr)
                else:
                    avg = std = summ = 0.
                    min_loc = max_loc = [0., 0]
            else:
                summ = comm.allreduce(np.array(val), op=mpi_ops["sum"])
                avg = summ / comm.Get_size()
                tmp = np.power(np.array(val - avg),2)
                std = comm.allreduce(tmp, op=mpi_ops["sum"])
                std = std / comm.Get_size()
                std = np.sqrt(std)
                min_loc = comm.allreduce((val,comm.Get_rank()), op=mpi_ops["min"])
                max_loc = comm.allreduce((val,comm.Get_rank()), op=mpi_ops["max"])
            stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
            }
            self.time_stats[key] = stats

    # Print timing statistics
    def print_stats(self, logger: logging.Logger):
        """Print timing statistics
        """
        for _, (key, val) in enumerate(self.time_stats.items()):
            stats_string = f": min = {val['min'][0]:>8e} , " + \
                           f"max = {val['max'][0]:>8e} , " + \
                           f"avg = {val['avg']:>8e} , " + \
                           f"std = {val['std']:>8e} "
            logger.info(f"SmartRedis {key} [s] " + stats_string)


##########################################################
##########################################################


# Dragon Client Class for Training
class Dragon_Train_Client:
    def __init__(self, cfg, rank: int, size: int):
        self.rank = rank
        self.size = size
        self._dd_serialized = cfg.online.dragon.dictionary
        self.launch = cfg.online.launch
        self.ppn = cfg.ppn
        self.ppd = cfg.ppd
        self.global_shuffling = cfg.online.global_shuffling
        self.batch = cfg.online.batch

        self._dd = None
        self.npts = None
        self.ndTot = None
        self.ndIn = None
        self.ndOut = None
        self.num_tot_tensors = None
        self.num_local_tensors = None
        self.sim_head_rank = None
        self.tensor_batch = None
        self.dataOverWr = None

        self.times = {
            "init": 0.,
            "tot_meta": 0.,
            "tot_train": 0.,
            "train": [],
        }
        self.time_stats = {}
        self.train_array_sz = 0

        if self.launch == "colocated":
            self.head_rank = self.rank//(self.ppn*self.ppd)
        elif self.launch == "clustered":
            self.head_rank = 0

    # Initialize client
    def init(self):
        """Initialize the client
        """
        tic = perf_counter()
        self._dd = DDict.attach(self._dd_serialized, timeout=30)
        toc = perf_counter()
        self.times['init'] = toc - tic

    # Destroy the client
    def destroy(self):
        """Destroy the client
        """
        self._dd.detach()

    # Check if tensor key exists
    def key_exists(self, key: str) -> bool:
        return key in self._dd.keys()

    # Get array (tensor) from DB
    def get_array(self, key: str, comm_type: str) -> np.ndarray:
        rtime = perf_counter()
        array = self._dd[key]
        rtime = perf_counter() - rtime
        if 'train' in comm_type:
            self.times["tot_train"] += rtime
            self.times["train"].append(rtime)
            self.train_array_sz = array.itemsize * array.size
        elif 'meta' in comm_type:
            self.times['tot_meta'] += rtime
        return array
    
    # Get value from Dictionary
    def get_value(self, key: str):
        rtime = perf_counter()
        val = self._dd[key][0]
        rtime = perf_counter() - rtime
        self.times['tot_meta'] += rtime
        return val
    
    # Put array (tensor) to DB
    def put_array(self, key: str, array: np.ndarray, comm_type: str) -> None:
        rtime = perf_counter()
        self._dd[key] = array
        rtime = perf_counter() - rtime
        if 'train' in comm_type:
            self.times["tot_train"] += rtime
            self.times["train"].append(rtime)
        elif 'meta' in comm_type:
            self.times['tot_meta'] += rtime

    # Put value to DB
    def put_value(self, key: str, value) -> None:
        rtime = perf_counter()
        self._dd[key] = np.array([value])
        rtime = perf_counter() - rtime
        self.times['tot_meta'] += rtime

    # Read the size information from DB
    def read_sizeInfo(self):
        while True:
            if (self.key_exists("sizeInfo")):
                dataSizeInfo = self.get_array('sizeInfo','meta')
                break
        self.npts = dataSizeInfo[0]
        self.ndTot = dataSizeInfo[1]
        self.ndIn = dataSizeInfo[2]
        self.ndOut = self.ndTot - self.ndIn
        self.num_tot_tensors = dataSizeInfo[3]
        self.num_local_tensors = dataSizeInfo[4]
        self.sim_head_rank = dataSizeInfo[5]
        
        max_batch_size = int(self.num_local_tensors/(self.ppn*self.ppd))
        if (not self.global_shuffling):
            self.tensor_batch = max_batch_size
        else:
            if (self.batch==0 or self.batch>max_batch_size):
                self.tensor_batch = max_batch_size
            else:
                self.tensor_batch = self.batch

    # Read the flag determining if data is overwritten in DB
    def read_overwrite(self):
        while True:
            if (self.key_exists("tensor-ow")):
                self.dataOverWr = self.get_value('tensor-ow')
                break

    # Set up the training problem from simulation meta data
    def setup_problem(self):
        self.read_sizeInfo()
        self.read_overwrite()
    
    # Check if model key exists
    def model_exists(self, key: str) -> bool:
        return self.key_exists(key)
    
    # Check delete model key
    def delete_model(self, key: str) -> None:
        del self._dd[key]

    # Put model to DB
    def put_model(self, key: str, model_bytes: io.BytesIO,
                  device: Optional[str] = None) -> None:
        self._dd[key] = model_bytes.getvalue()
        

    # Collect timing statistics across ranks
    def collect_stats(self, comm):
        for _, (key, val) in enumerate(self.times.items()):
            if (key=="train"):
                collected_arr = np.zeros((len(val)*comm.size))
                comm.comm.Gather(np.array(val),collected_arr,root=0)
                avg = np.mean(collected_arr)
                std = np.std(collected_arr)
                min = np.amin(collected_arr); min_loc = [min, 0]
                max = np.amax(collected_arr); max_loc = [max, 0]
                summ = np.sum(collected_arr)
            else:
                summ = comm.comm.allreduce(np.array(val), op=comm.sum)
                avg = summ / comm.size
                tmp = np.power(np.array(val - avg),2)
                std = comm.comm.allreduce(tmp, op=comm.sum)
                std = std / comm.size
                std = np.sqrt(std)
                min_loc = comm.comm.allreduce((val,comm.rank), op=comm.minloc)
                max_loc = comm.comm.allreduce((val,comm.rank), op=comm.maxloc)
            stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
            }
            self.time_stats[key] = stats

    # Print timing statistics
    def print_stats(self, logger: logging.Logger):
        """Print timing statistics
        """
        for _, (key, val) in enumerate(self.time_stats.items()):
            stats_string = f": min = {val['min'][0]:>8e} , " + \
                           f"max = {val['max'][0]:>8e} , " + \
                           f"avg = {val['avg']:>8e} , " + \
                           f"std = {val['std']:>8e} "
            logger.info(f"SmartRedis {key} [s] " + stats_string)

