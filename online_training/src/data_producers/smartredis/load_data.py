from argparse import ArgumentParser
from math import pi as PI
from time import perf_counter, sleep
from typing import Tuple, Optional
import numpy as np

from smartredis import Client

# SmartRedis Client Class
class SmartRedisClient:
    def __init__(self, args, rank: int, size: int):
        self.client = None
        self.db_launch = args.db_launch
        self.db_nodes = args.db_nodes
        self.rank = rank
        self.size = size
        self.ppn = args.ppn
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

    # Initialize client
    def init_client(self, comm):
        if (self.db_nodes==1):
            tic = perf_counter()
            self.client = Client(cluster=False)
            toc = perf_counter()
        else:
            tic = perf_counter()
            self.client = Client(cluster=True)
            toc = perf_counter()
        self.times["init"] = toc - tic
        comm.Barrier()
        if (self.rank==0):
            print('All SmartRedis clients initialized \n', flush=True)
    
    # Set up training case and write metadata
    def setup(self, comm, n_samples: int, ndTot: int, ndIn: int):
        if (self.rank%self.ppn == 0):
            # Run-check
            arr = np.array([1, 1], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('check-run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

            # Training data setup
            dataSizeInfo = np.empty((6,), dtype=np.int64)
            dataSizeInfo[0] = n_samples
            dataSizeInfo[1] = ndTot
            dataSizeInfo[2] = ndIn
            dataSizeInfo[3] = comm.Get_size()
            dataSizeInfo[4] = self.ppn
            dataSizeInfo[5] = self.head_rank
            tic = perf_counter()
            self.client.put_tensor('sizeInfo', dataSizeInfo)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

            # Write overwrite tensor
            tic = perf_counter()
            self.client.put_tensor('tensor-ow', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

        comm.Barrier()
        if (self.rank==0):
            print('Metadata sent to DB \n', flush=True)

    # Check if should keep running
    def check_run(self) -> bool:
        tic = perf_counter()
        arr = self.client.get_tensor('check-run')
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
        if (arr[0]==0):
            return True
        else:
            return False
    
    # Signal to training sim is exiting
    def stop_train(self, comm):
        if (self.rank%self.ppn == 0):
            # Run-check
            arr = np.array([0, 0], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('check-run', arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic

        comm.Barrier()
        if (self.rank==0):
            print('Told training to exit \n', flush=True)
        
    # Send training snapshot
    def send_snapshot(self, array: np.ndarray, step: int):
        key = 'y.'+str(self.rank) #+'.'+str(step)
        if (self.rank==0):
            print(f'Sending training data with key {key} and shape {array.shape}', flush=True)
        tic = perf_counter()
        self.client.put_tensor(key, array)
        toc = perf_counter()
        self.times["tot_train"] += toc - tic
        self.times["train"].append(toc - tic)

    # Send time step
    def send_step(self, step: int):
        if (self.rank%self.ppn == 0):
            step_arr = np.array([step, step], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('step', step_arr)
            toc = perf_counter()
            self.times["tot_meta"] += toc - tic
    
    # Check to see if model exists in DB
    def model_exists(self, comm, model_name: str) -> bool:
        tic = perf_counter()
        local_exists = 1 if self.client.model_exists(model_name) else 0
        toc = perf_counter()
        self.times["tot_meta"] += toc - tic
        global_exists = comm.allreduce(local_exists)
        if global_exists==self.size:
            if self.rank == 0:
                print("\nFound model checkpoint in DB\n", flush=True)
            return True
        else:
            return False
        
    # Perform inference with model on DB
    def infer_model(self, comm, model_name: str, inputs: np.ndarray,
                    outputs: np.ndarray) -> float:
        input_key = f"{model_name}_inputs_{self.rank}"
        output_key = f"{model_name}_outputs_{self.rank}"
        if inputs.ndim<2:
            inputs = np.expand_dims(inputs, axis=1)
        tic = perf_counter()
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

# Generate training data for each model
def generate_training_data(args, rank: int, step: Optional[int] = 0) -> Tuple[np.ndarray, dict]:
    """Generate training data for each model
    """
    random_seed = 12345 + 1000*rank + 100*step
    rng = np.random.default_rng(seed=random_seed)
    # For the MLP model, train versions of y=sin(x)
    if (args.model=="mlp"):
        if (args.problem_size=="small"):
            n_samples = 512
            ndIn = 1
            ndTot = 2
            x = rng.uniform(low=0.0, high=2*PI, size=n_samples)
            y = np.sin(x)+0.1*np.sin(4*PI*x)
            y = (y - (-1.0875)) / (1.0986 - (-1.0875)) # min-max scaling
            data = np.vstack((x,y)).T

    return_dict = {
        "n_samples": n_samples,
        "n_dim_in": ndIn,
        "n_dim_tot": ndTot
    }
    return data, return_dict

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

# Main data producer function
def main():
    """Emulate a data producing simulation for online training with SmartSim/SmartRedis
    """
    # MPI Init
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    comm.Barrier()

    # Parse arguments
    parser = ArgumentParser(description='SmartRedis Data Producer')
    parser.add_argument('--model', default="mlp", type=str, help='ML model identifier (mlp, quadconv, gnn)')
    parser.add_argument('--problem_size', default="small", type=str, help='Size of problem to emulate (small, medium, large)')
    parser.add_argument('--tolerance', default=0.01, type=float, help='ML model convergence tolerance')
    parser.add_argument('--db_launch', default="colocated", type=str, help='Database deployment (colocated, clustered)')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of database nodes')
    parser.add_argument('--ppn', default=4, type=int, help='Number of processes per node')
    parser.add_argument('--logging', default='no', help='Level of performance logging (no, verbose)')
    parser.add_argument('--reproducibility', type=str, default='False', help='Send a single array for reproducible results')
    args = parser.parse_args()

    rankl = rank % args.ppn
    if (rank==0 and args.logging=="verbose"):
        print(f"Hello from MPI rank {rank}/{size}, local rank {rankl} and node {name}")
    if (rank==0):
        print(f'\nRunning with {args.db_nodes} DB nodes', flush=True)
        print(f'and with {args.ppn} processes per node \n', flush=True)

    # Initialize SmartRedis clients
    client = SmartRedisClient(args, rank, size)
    client.init_client(comm)

    # Generate synthetic data for the specific model
    train_array, stats = generate_training_data(args, rank)

    # Send training metadata
    client.setup(comm, stats["n_samples"], 
                 stats["n_dim_tot"], stats["n_dim_in"])

    # Emulate integration of PDEs with a do loop
    numts = 1000
    success = 0
    tic_loop = perf_counter()
    for step in range(numts):
        # Sleep for a few seconds to emulate the time required by PDE integration
        sleep(1)
        if args.reproducibility=="False":
            train_array, _ = generate_training_data(args, rank, step)

        # Check if model exists to perform inference
        exists = client.model_exists(comm, args.model)
        if exists:
            if (args.problem_size=="small"):
                inputs = train_array[:,0]
                outputs = train_array[:,1]
            error = client.infer_model(comm, args.model, inputs, outputs)
            if (rank==0):
                print(f"Performed inference with error={error:>8e}", flush=True)
            if error <= args.tolerance:
                success += 1
            else:
                success = 0

        # Send training data
        client.send_snapshot(train_array, step)
        comm.Barrier()
        if (rank==0):
            print(f'All ranks finished sending training data', flush=True)
        client.send_step(step)

        # Exit if model has converged to tolerence for 5 consecutive checks
        if success>=5: 
            if rank==0:
                print("\nModel has converged to tolerence for 5 consecutive checks", flush=True)
            client.stop_train(comm)
            break

    toc_loop = perf_counter()
    time_to_solution = toc_loop - tic_loop

    # Accumulate timing data for client and print summary
    if rank==0:
        print("Summary of timing data:", flush=True)
    mpi_ops = {
        "sum": MPI.SUM,
        "min": MPI.MINLOC,
        "max": MPI.MAXLOC
    }
    client.collect_stats(comm, mpi_ops)
    if (rank==0):
        client.print_stats()

    # Print FOM
    train_array_sz = train_array.itemsize * train_array.size / 1024 / 1024 / 1024
    if rank==0:
        print("\nFOM:")
        print_fom(time_to_solution, train_array_sz, client.time_stats)


if __name__ == "__main__":
    main()