import sys
import os, os.path
from typing import Optional
from time import perf_counter
import logging
import numpy as np
import torch
try:
    from torch_geometric.nn import knn_graph
except:
    pass

from smartredis import Client

# SmartRedis Client Class for the Simulation (Data Producer)
class SmartRedis_Sim_Client:
    def __init__(self, args, rank: int, size: int):
        self.client = None
        self.db_launch = args.db_launch
        self.db_nodes = args.db_nodes
        self.rank = rank
        self.size = size
        self.ppn = args.ppn
        self.ow = True if args.problem_size=="debug" else False
        self.max_mem = args.db_max_mem_size*1024*1024*1024
        self.db_address = None
        self.model = args.model

        self.times = {
            "init": 0.,
            "tot_meta": 0.,
            "tot_train": 0.,
            "train": [],
            "tot_infer": 0.,
            "infer": []
        }
        self.time_stats = {}

        if (self.db_launch == "colocated"):
            self.db_nodes = 1
            self.head_rank = self.ppn * self.rank/self.ppn
        elif (self.db_launch == "clustered"):
            self.ppn = size
            self.head_rank = 0

        # For GNN inference
        self.coords = None
        self.edge_index = None

    # Initialize client
    def init_client(self):
        self.db_address = os.environ["SSDB"]
        tic = perf_counter()
        if (self.db_nodes==1):
            self.client = Client(cluster=False)
        else:
            self.client = Client(cluster=True)
        toc = perf_counter()
        self.times["init"] = toc - tic
    
    # Set up training case and write metadata
    def setup_training_problem(self, coords: np.ndarray, 
                               data_info: dict) -> None:
        if (self.rank%self.ppn == 0):
            # Run-check
            arr = np.array([1], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('sim-run', arr)
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
            self.client.put_tensor('sizeInfo', dataSizeInfo)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

            # Write overwrite tensor
            if self.ow:
                arr = np.array([1], dtype=np.int64)
            else:
                arr = np.array([0], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('tensor-ow', arr)
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
        self.client.put_tensor(f'pos_node_{self.rank}', self.coords)
        self.client.put_tensor(f'edge_index_{self.rank}', self.edge_index)
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
    
    # Signal to training sim is exiting
    def stop_train(self):
        if (self.rank%self.ppn == 0):
            arr = np.array([0], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('sim-run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
        
    # Send training snapshot
    def send_snapshot(self, array: np.ndarray, step: int):
        if self.ow:
            key = 'x.'+str(self.rank)
        else:
            key = 'x.'+str(self.rank)+'.'+str(step)
        tic = perf_counter()
        self.client.put_tensor(key, array)
        toc = perf_counter()
        self.times["tot_train"] += toc - tic
        self.times["train"].append(toc - tic)

    # Check DB memory
    def check_db_mem(self, array: np.ndarray) -> bool:
        tic = perf_counter()
        db_info=self.client.get_db_node_info([self.db_address])[0]
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
        used_mem = float(db_info['Memory']['used_memory'])
        free_mem = self.max_mem - used_mem
        if (sys.getsizeof(array) < free_mem):
            return True
        else:
            return False

    # Send time step
    def send_step(self, step: int):
        if (self.rank%self.ppn == 0):
            step_arr = np.array([step], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('step', step_arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
    
    # Check to see if model exists in DB
    def model_exists(self, comm) -> bool:
        tic = perf_counter()
        if (self.model=='gnn'):
            local_exists = 1 if os.path.exists(f"/tmp/{self.model}.pt") else 0
        else:
            local_exists = 1 if self.client.model_exists(self.model) else 0
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
        global_exists = comm.allreduce(local_exists)
        if global_exists==self.size:
            #if self.rank == 0:
            #    print("\nFound model checkpoint in DB\n", flush=True)
            return True
        else:
            return False
        
    # Perform inference with model on DB
    def infer_model(self, comm, inputs: np.ndarray,
                    outputs: np.ndarray) -> float:
        if inputs.ndim<2:
            inputs = np.expand_dims(inputs, axis=1)
        tic = perf_counter()
        if (self.model=="gnn"):
            #model_bytes = self.client.get_tensor(model_name)[0]
            #buffer = io.BytesIO(model_bytes)
            #model_jit = torch.jit.load(buffer)
            while True:
                try:
                    model_jit = torch.jit.load(f"/tmp/{self.model}.pt")
                    break
                except:
                    pass
            x = torch.from_numpy(inputs).type(torch.float32)
            edge_index = torch.from_numpy(self.edge_index).type(torch.int64)
            pos = torch.from_numpy(self.coords).type(torch.float32)
            with torch.no_grad():
                pred = model_jit(x, edge_index, pos).numpy()
        else:
            input_key = f"{self.model}_inputs_{self.rank}"
            output_key = f"{self.model}_outputs_{self.rank}"
            self.client.put_tensor(input_key, inputs.astype(np.float32))
            self.client.run_model(self.model, inputs=[input_key], 
                                  outputs=[output_key])
            pred = self.client.get_tensor(output_key)
        toc = perf_counter()
        self.times["tot_infer"] += toc - tic
        self.times["infer"].append(toc - tic)
        local_mse = ((outputs.flatten() - pred.flatten())**2).mean()
        avg_mse = comm.allreduce(local_mse)/self.size
        return avg_mse
    
    # Check status of training
    def check_train_status(self) -> None:
        while True:
            if (self.client.poll_tensor('train-run',0,1)):
                break

    # Collect timing statistics across ranks
    def collect_stats(self, comm, mpi_ops):
        for _, (key, val) in enumerate(self.times.items()):
            if (key=="train" or key=="infer"):
                collected_arr = np.zeros((len(val)*comm.Get_size()))
                comm.Gather(np.array(val),collected_arr,root=0)
                avg = np.mean(collected_arr)
                std = np.std(collected_arr)
                min = np.amin(collected_arr); min_loc = [min, 0]
                max = np.amax(collected_arr); max_loc = [max, 0]
                summ = np.sum(collected_arr)
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


# SmartRedis Client Class for Training
class SmartRedis_Train_Client:
    def __init__(self, cfg, rank: int, size: int):
        self.rank = rank
        self.size = size
        self.db_launch = cfg.online.smartredis.db_launch
        self.db_nodes = cfg.online.smartredis.db_nodes
        self.ppn = cfg.ppn
        self.ppd = cfg.ppd
        self.global_shuffling = cfg.online.global_shuffling
        self.batch = cfg.online.batch

        self.client = None
        self.npts = None
        self.ndTot = None
        self.ndIn = None
        self.ndOut = None
        self.num_tot_tensors = None
        self.num_db_tensors = None
        self.head_rank = None
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

    # Initializa client
    def init(self):
        """Initialize the SmartRedis client
        """
        # Read the address of the co-located database first
        if (self.db_launch=='colocated'):
            address = os.environ['SSDB']
        else:
            address = None

        # Initialize Redis clients on each rank 
        cluster = False if self.db_nodes==1 else True
        tic = perf_counter()
        self.client = Client(address=address, cluster=cluster)
        toc = perf_counter()
        self.times['init'] = toc - tic

    # Check if tensor key exists
    def key_exists(self, key: str) -> bool:
        return self.client.poll_tensor(key,0,1)

    # Get array (tensor) from DB
    def get_array(self, key: str, comm_type: str) -> np.ndarray:
        rtime = perf_counter()
        array = self.client.get_tensor(key)
        rtime = perf_counter() - rtime
        if 'train' in comm_type:
            self.times["tot_train"] += rtime
            self.times["train"].append(rtime)
            self.train_array_sz = array.itemsize * array.size
        elif 'meta' in comm_type:
            self.times['tot_meta'] += rtime
        return array
    
    # Get value from DB
    def get_value(self, key: str):
        rtime = perf_counter()
        val = self.client.get_tensor(key)[0]
        rtime = perf_counter() - rtime
        self.times['tot_meta'] += rtime
        return val
    
    # Put array (tensor) to DB
    def put_array(self, key: str, array: np.ndarray, comm_type: str) -> None:
        rtime = perf_counter()
        self.client.put_tensor(key, array)
        rtime = perf_counter() - rtime
        if 'train' in comm_type:
            self.times["tot_train"] += rtime
            self.times["train"].append(rtime)
        elif 'meta' in comm_type:
            self.times['tot_meta'] += rtime

    # Put value to DB
    def put_value(self, key: str, value) -> None:
        rtime = perf_counter()
        self.client.put_tensor(key, np.array([value]))
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
        self.num_db_tensors = dataSizeInfo[4]
        self.head_rank = dataSizeInfo[5]
        
        max_batch_size = int(self.num_db_tensors/(self.ppn*self.ppd))
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
        if (self.rank==0):
            if (self.dataOverWr>0.5): 
                print("\nTraining data is overwritten in DB \n")
            else:
                print("\nTraining data is accumulated in DB \n")
            sys.stdout.flush()

    # Set up the training problem from simulation meta data
    def setup_problem(self):
        self.read_sizeInfo()
        self.read_overwrite()
    
    # Check if model key exists
    def model_exists(self, key: str) -> bool:
        return self.client.model_exists(key)
    
    # Check delete model key
    def delete_model(self, key: str) -> None:
        self.client.delete_model(key)

    # Put model to DB
    def put_model(self, key: str, model_bytes: bytes,
                  ml_framework: Optional[str] = 'TORCH',
                  device: Optional[str] = 'CPU') -> None:
        self.client.set_model(key, model_bytes, ml_framework, device)

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
