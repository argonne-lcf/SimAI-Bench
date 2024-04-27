import sys
import os, os.path
from time import perf_counter
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
        if (self.db_nodes==1):
            tic = perf_counter()
            self.client = Client(cluster=False)
            toc = perf_counter()
        else:
            tic = perf_counter()
            self.client = Client(cluster=True)
            toc = perf_counter()
        self.times["init"] = toc - tic
    
    # Set up training case and write metadata
    def setup_training_problem(self, data_info: dict):
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

    # Set up training case and write metadata
    def setup_graph(self, coords: np.ndarray, rank: int):
        if coords.ndim<2:
            coords = np.expand_dims(coords, axis=1)
        self.coords = coords
        self.edge_index = knn_graph(torch.from_numpy(coords), k=2, loop=False).numpy()
        tic = perf_counter()
        self.client.put_tensor(f'pos_node_{rank}', self.coords)
        self.client.put_tensor(f'edge_index_{rank}', self.edge_index)
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
    
    # Signal to training sim is exiting
    def stop_train(self, comm):
        if (self.rank%self.ppn == 0):
            # Run-check
            arr = np.array([0], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('sim-run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

        comm.Barrier()
        if (self.rank==0):
            print('Told training to exit \n', flush=True)
        
    # Send training snapshot
    def send_snapshot(self, array: np.ndarray, step: int):
        if self.ow:
            key = 'x.'+str(self.rank)
        else:
            key = 'x.'+str(self.rank)+'.'+str(step)
        if (self.rank==0):
            print(f'\tSending training data with key {key} and shape {array.shape}', flush=True)
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
    def model_exists(self, comm, model_name: str) -> bool:
        tic = perf_counter()
        if (model_name=="gnn"):
            local_exists = 1 if os.path.exists(f"/tmp/{model_name}.pt") else 0
        else:
            local_exists = 1 if self.client.model_exists(model_name) else 0
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
    def infer_model(self, comm, model_name: str, inputs: np.ndarray,
                    outputs: np.ndarray) -> float:
        if inputs.ndim<2:
            inputs = np.expand_dims(inputs, axis=1)
        tic = perf_counter()
        if (model_name=="gnn"):
            #model_bytes = self.client.get_tensor(model_name)[0]
            #buffer = io.BytesIO(model_bytes)
            #model_jit = torch.jit.load(buffer)
            while True:
                try:
                    model_jit = torch.jit.load(f"/tmp/{model_name}.pt")
                    break
                except:
                    pass
            x = torch.from_numpy(inputs).type(torch.float32)
            edge_index = torch.from_numpy(self.edge_index).type(torch.int64)
            pos = torch.from_numpy(self.coords).type(torch.float32)
            with torch.no_grad():
                pred = model_jit(x, edge_index, pos).numpy()
        else:
            input_key = f"{model_name}_inputs_{self.rank}"
            output_key = f"{model_name}_outputs_{self.rank}"
            self.client.put_tensor(input_key, inputs.astype(np.float32))
            self.client.run_model(model_name, inputs=[input_key], 
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
    def print_stats(self):
        """Print timing statistics
        """
        for _, (key, val) in enumerate(self.time_stats.items()):
            stats_string = f": min = {val['min'][0]:>8e} , " + \
                           f"max = {val['max'][0]:>8e} , " + \
                           f"avg = {val['avg']:>8e} , " + \
                           f"std = {val['std']:>8e} "
            print(f"SmartRedis {key} [s] " + stats_string)


# SmartRedis Client Class for Training
class SmartRedis_Train_Client:
    def __init__(self):
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

    # Initializa client
    def init(self, cfg, comm, t_data):
        """Initialize the SmartRedis client
        """
        try:
            from smartredis import Client
        except ModuleNotFoundError as err:
            if comm.rank==0: print(err)

        # Read the address of the co-located database first
        if (cfg.online.smartredis.db_launch=='colocated'):
            #prefix = f'{cfg.online.simprocs}-procs_case/'
            #address = self.read_SSDB(prefix, comm)
            address = os.environ['SSDB']
        else:
            address = None

        # Initialize Redis clients on each rank #####
        if (comm.rank == 0):
            print("\nInitializing Python clients ...", flush=True)
        if (cfg.online.smartredis.db_nodes==1):
            rtime = perf_counter()
            sys.stdout.flush()
            self.client = Client(address=address, cluster=False)
            rtime = perf_counter() - rtime
            t_data.t_init = t_data.t_init + rtime
            t_data.i_init = t_data.i_init + 1
        else:
            rtime = perf_counter()
            self.client = Client(address=address, cluster=True)
            rtime = perf_counter() - rtime
            t_data.t_init = t_data.t_init + rtime
            t_data.i_init = t_data.i_init + 1
        comm.comm.Barrier()
        if (comm.rank == 0):
            print("All done\n", flush=True)

    # Read the address of the co-located database
    def read_SSDB(self, prefix, comm):
        SSDB_file = prefix + f'SSDB_{comm.name}.dat'
        c = 0 
        while True:
            if (exists(SSDB_file)):
                f = open(SSDB_file, "r")
                SSDB = f.read()
                f.close()
                if (SSDB == ''):
                    continue
                else:
                    print(f'[{comm.rank}]: read SSDB={SSDB}')
                    sys.stdout.flush()
                    break
            else:
                if (c==0):
                    print(f'[{comm.rank}]: WARNING, looked for {SSDB_file} but did not find it')
                    sys.stdout.flush()
                c+=1
                continue
        comm.comm.Barrier()
        if ('\n' in SSDB):
            SSDB = SSDB.replace('\n', '') 
        return SSDB

    # Read the size information from DB
    def read_sizeInfo(self, cfg, comm, t_data):
        if (comm.rank == 0):
            print("\nGetting size info from DB ...")
            sys.stdout.flush()
        while True:
            if (self.client.poll_tensor("sizeInfo",0,1)):
                rtime = perf_counter()
                dataSizeInfo = self.client.get_tensor('sizeInfo')
                rtime = perf_counter() - rtime
                t_data.t_meta = t_data.t_meta + rtime
                t_data.i_meta = t_data.i_meta + 1
                break
        self.npts = dataSizeInfo[0]
        self.ndTot = dataSizeInfo[1]
        self.ndIn = dataSizeInfo[2]
        self.ndOut = self.ndTot - self.ndIn
        self.num_tot_tensors = dataSizeInfo[3]
        self.num_db_tensors = dataSizeInfo[4]
        self.head_rank = dataSizeInfo[5]
        
        max_batch_size = int(self.num_db_tensors/(cfg.ppn*cfg.ppd))
        if (not cfg.online.global_shuffling):
            self.tensor_batch = max_batch_size
        else:
            if (cfg.online.batch==0 or cfg.online.batch>max_batch_size):
                self.tensor_batch = max_batch_size
            else:
                self.tensor_batch = cfg.online.batch

        if (comm.rank == 0):
            print(f"Samples per simulation tensor: {self.npts}")
            print(f"Model input features: {self.ndIn}")
            print(f"Model output targets: {self.ndOut}")
            print(f"Total tensors in all DB: {self.num_tot_tensors}")
            print(f"Tensors in local DB: {self.num_db_tensors}")
            print(f"Simulation tensors per batch: {self.tensor_batch}")
            sys.stdout.flush()

    # Read the flag determining if data is overwritten in DB
    def read_overwrite(self, comm, t_data):
        while True:
            if (self.client.poll_tensor("tensor-ow",0,1)):
                rtime = perf_counter()
                self.dataOverWr = self.client.get_tensor('tensor-ow')
                rtime = perf_counter() - rtime
                t_data.t_meta = t_data.t_meta + rtime
                t_data.i_meta = t_data.i_meta + 1 
                break
        if (comm.rank==0):
            if (self.dataOverWr>0.5): 
                print("\nTraining data is overwritten in DB \n")
            else:
                print("\nTraining data is accumulated in DB \n")
            sys.stdout.flush()
