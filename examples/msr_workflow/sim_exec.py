import mpi4py
mpi4py.rc.initialize=False
from mpi4py import MPI
from SimAIBench.simulation import Simulation as sim
import argparse
import numpy as np
import logging as logging_
import time
import json
import os
import pyitt
from typing import Union

def compute_mean_std(comm, data, counts):
    mask = counts > 0
    count = np.sum(counts)
    count = comm.allreduce(count, op=MPI.SUM)
    mean = np.sum(data[mask])
    mean = comm.allreduce(mean, op=MPI.SUM) / count
    std = np.sum((data[mask]/counts[mask] - mean) ** 2)
    std = comm.allreduce(std, op=MPI.SUM) / count
    std = np.sqrt(std)
    return mean, std

def compute_mean_std_dragon(ddict, data, counts, my_rank, metric, size):
    mask = counts > 0
    count = np.sum(counts)
    mean = np.sum(data[mask])
    ##compute global mean first
    global_mean = 0.0

    all_means = np.zeros(size, dtype=np.float64)
    all_counts = np.zeros(size, dtype=np.int32)
    ####
    key = f"{my_rank}_meanCount_{metric}"
    ddict.pput(key, (mean,count))
    all_means[my_rank] = mean
    all_counts[my_rank] = count
    done_ranks = set([my_rank])
    for retry in range(100):
        if len(done_ranks) == size:
            break
        for rank in range(size):
            key = f"{rank}_meanCount_{metric}"
            if key in ddict.keys() and rank not in done_ranks:
                all_means[rank], all_counts[rank] = ddict[key]
                done_ranks.add(rank)
            
    # ###put counts
    # all_counts = np.zeros(size, dtype=np.int32)
    # all_counts[my_rank] = count
    # key = f"{my_rank}_count_{metric}"
    # ddict.pput(key, count)
    # done_ranks = set([my_rank])
    # for retry in range(100):
    #     if len(done_ranks) == size:
    #         break
    #     for rank in range(size):
    #         key = f"{rank}_count_{metric}"
    #         if key in ddict.keys() and rank not in done_ranks:
    #             all_counts[rank] = ddict[key]
    #             done_ranks.add(rank)
    # # while len(done_ranks) < size:
    # #     for key in ddict.keys():
    # #         if metric not in key:
    # #             continue
    # #         rank, stat, metric = key.split("_")
    # #         rank = int(rank)
    # #         if stat == "count":
    # #             all_counts[rank] = ddict.bget(key)
    # #             done_ranks.add(rank)
    global_mean = np.sum(all_means) / np.sum(all_counts)

    std = np.sum((data[mask]/counts[mask] - global_mean) ** 2)
    key = f"{my_rank}_stdCount_{metric}"
    ddict.pput(key, (std,count))
    all_stds = np.zeros(size, dtype=np.float64)
    all_counts = np.zeros(size, dtype=np.int32)
    all_stds[my_rank] = std
    all_counts[my_rank] = count
    done_ranks = set([my_rank])
    for retry in range(100):
        if len(done_ranks) == size:
            break
        for rank in range(size):
            key = f"{rank}_stdCount_{metric}"
            if key in ddict.keys() and rank not in done_ranks:
                all_stds[rank], all_counts[rank] = ddict[key]
                done_ranks.add(rank)
    # while len(done_ranks) < size:
    #     for key in ddict.keys():
    #         if metric not in key:
    #             continue
    #         rank, stat, metric = key.split("_")
    #         rank = int(rank)
    #         if stat == "std":
    #             all_stds[rank] = ddict.bget(key)
    #             done_ranks.add(rank)
    global_std = np.sqrt(np.sum(all_stds) / np.sum(all_counts))
    
    return global_mean, global_std

def main(sim_config:dict,
         server_info:Union[str, dict]=None,
         sim_id:int=0,
         is_colocated:bool=False,
         init_MPI:bool=True,
         rank:int=0,
         size:int=1,
         dtype=np.float32,
         init_time:float=0.0):
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    ##
    data_size = sim_config["data_size"]
    write_freq = sim_config["write_freq"]
    sim_telemetry = sim_config["sim_telemetry"]
    nrequests = sim_config.get("nrequests", 1)
    nsteps = sim_config.get("nsteps", 1000)

    # Create a simulation object
    simulation = sim(name=f"sim_{sim_id}_{rank}", 
                     comm=(comm if init_MPI else None), 
                     server_info=server_info,
                     logging=rank == 0,  # Only the root rank logs
                     log_level=logging_.DEBUG,
                     size=size, 
                     rank=rank,
                     is_colocated=is_colocated)
    # Initialize the simulation from a JSON file
    simulation.init_from_json(os.path.join(os.path.dirname(__file__), sim_telemetry))
    if simulation.logger:
        simulation.logger.info(f"Simulation initialized with name {simulation.name}, rank {rank}, size {size}")
    if init_MPI:
        comm.Barrier()
    i=0
    warmup_time = time.time()
    simulation.run()
    warmup_time = time.time() - warmup_time
    if warmup_time < init_time:
        time.sleep(init_time-warmup_time) 
    # Prepare arrays for timing
    iteration_time_array = np.zeros(nsteps, dtype=np.float32)
    data_write_time_array = np.zeros(nsteps, dtype=np.float32)
    for step in range(nsteps):
        tic = time.time()
        # Run the simulation step
        if os.getenv("PROFILE_WORKFLOW", None) is not None:
            with pyitt.task(f"simulation_step_{step}",domain="simulation"):
                iter_dt_out = simulation.run(nsteps=1)
        else:
            iter_dt_out = simulation.run(nsteps=1)
        iter_time = time.time() - tic
        # Stage data for the AI to read
        if step % write_freq == 0:
            fname_ext = f"input_{sim_id}"
            if simulation.logger:
                simulation.logger.info(f"Write the data: {fname_ext}_{rank}_{step//write_freq}")
            tic = time.time()
            simulation.stage_write(f"{fname_ext}_{rank}_{step//write_freq}", np.empty(data_size, dtype=dtype))
            toc = time.time()
            data_write_time = toc - tic
        else:
            data_write_time = 0.0
        
        # Store timing data
        iteration_time_array[step] = iter_time
        data_write_time_array[step] = data_write_time

        if simulation.logger:
            simulation.logger.info(f"tstep time: {iter_time}, dt time: {data_write_time}")

    if server_info["config"]["type"] != "dragon" and comm is not None:
        counts = (iteration_time_array > 0).astype(np.int32)
        mean,std = compute_mean_std(comm, iteration_time_array, counts)
        if simulation.logger:
            simulation.logger.info(f"Mean iteration time (s): {mean}, Std: {std}")
        counts = (data_write_time_array > 0).astype(np.int32)
        mean,std = compute_mean_std(comm, data_write_time_array, counts)
        if simulation.logger:
            simulation.logger.info(f"Mean data write time (s): {mean}, Std: {std}")
        
        throughput = np.zeros_like(data_write_time_array)
        mask = data_write_time_array > 0
        counts = mask.astype(np.int32)
        throughput[mask] = np.prod(data_size)*1e-6*dtype().itemsize/data_write_time_array[mask]
        mean,std = compute_mean_std(comm, throughput, counts)
        if simulation.logger:
            simulation.logger.info(f"Mean write throughput (MB/s): {mean}, Std: {std}")
    elif server_info["config"]["type"] =="dragon":
        ##barrier
        # ndone = [simulation.name]
        # simulation.global_ddict.pput(simulation.name,"done")
        # while len(ndone) < size:
        #     for i in range(size):
        #         key = f"sim_{i}"
        #         if key in ndone:
        #             continue
        #         if key in simulation.global_ddict.keys():
        #             if simulation.global_ddict[key] == "done":
        #                 ndone.append(key)
        #     time.sleep(0.5)
        
        ddict = simulation.global_ddict
        mean, std = compute_mean_std_dragon(ddict, iteration_time_array, (iteration_time_array > 0).astype(np.int32), rank, "simIterationTime", size)
        if simulation.logger:
            simulation.logger.info(f"Mean iteration time (s): {mean}, Std: {std}")
        mean, std = compute_mean_std_dragon(ddict, data_write_time_array, (data_write_time_array > 0).astype(np.int32), rank, "dataWriteTime", size)
        if simulation.logger:
            simulation.logger.info(f"Mean data write time (s): {mean}, Std: {std}")
        throughput = np.zeros_like(data_write_time_array)
        mask = data_write_time_array > 0
        counts = mask.astype(np.int32)
        throughput[mask] = np.prod(data_size)*1e-6*dtype().itemsize/data_write_time_array[mask]
        mean, std = compute_mean_std_dragon(ddict, throughput, counts, rank, "writeThroughput", size)
        if simulation.logger:
            simulation.logger.info(f"Mean write throughput (MB/s): {mean}, Std: {std}")

    simulation.flush_logger()
    # simulation.clean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Sim config")
    parser.add_argument("--server_info", type=str, default=None, help="Server info for the simulation")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config), "r") as f:
        config = json.load(f)

    main(config, server_info=args.server_info)
