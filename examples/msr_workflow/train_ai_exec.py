
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse
import logging as logging_
import json
import socket
import os
import pyitt
import numpy as np
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

def main(ai_config:str,
         server_info:Union[str, dict]=None,
         nsims:int=1,
         is_colocated:bool=False,
         init_MPI=True,
         rank:int=0,
         size:int=1,
         init_time:float=0.0):
    device = ai_config["device"]
    nsteps = ai_config["num_iters"]
    update_frequency = ai_config["read_freq"]
    run_time = ai_config["run_time"]
    nrequests = ai_config.get("nrequests", 1)
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    train_ai = AI(f"train_ai_{rank}", server_info=server_info, num_hidden_layers=1, num_epochs=1, 
                    device=device,logging=rank == 0,log_level=logging_.DEBUG,data_size=2, is_colocated=is_colocated)
    if train_ai.logger:
        train_ai.logger.info(f"AI initialized with name {train_ai.name}, rank {rank}, size {size}, nsims {nsims}")
        
    if init_MPI:
        comm.Barrier()
    
    warmup_time = time.time()
    train_ai.train(run_time=run_time)
    warmup_time = time.time() - warmup_time
    if warmup_time < init_time:
        time.sleep(init_time-warmup_time)

    # Prepare arrays for timing
    iteration_time = np.zeros(nsteps, dtype=np.float32)
    data_read_time = np.zeros(nsteps, dtype=np.float32)
    read_count = np.zeros(nsteps, dtype=np.int32)

    for step in range(nsteps):
        tic = time.time()
        if os.getenv("PROFILE_WORKFLOW",None) is not None:
            with pyitt.task(f"inference_step_{step}",domain="training"):
                elap_time, rc = train_ai.train(run_time=run_time)
        else:
            elap_time, rc = train_ai.train(run_time=run_time)
        train_time = time.time() - tic
        
        if train_ai.config["type"] == "redis":
            nclients = len(train_ai.redis_client)
        elif train_ai.config["type"] == "dragon":
            nclients = len(train_ai.dragon_dict.manager_nodes)
        else:
            nclients = 1
            
        ##comsume the simulation data
        if step%update_frequency == 0:
            tstart = time.time()
            dt_time = 0.0
            read_sims = set()
            while len(read_sims) < nsims:
                if train_ai.logger:
                    train_ai.logger.info(f"Waiting for data to be staged for step {step} remaining sims {[sim_id for sim_id in range(nsims) if sim_id not in read_sims]}...")
                for sim_id in range(nsims):
                    try:
                        for client_id in range(nclients):
                            if sim_id not in read_sims and train_ai.poll_staged_data(f"input_{sim_id}_{rank}_{step//update_frequency}",client_id=client_id,is_local=True):
                                tic = time.time()
                                data = train_ai.stage_read(f"input_{sim_id}_{rank}_{step//update_frequency}",client_id=client_id,is_local=True)
                                dt_time += time.time() - tic
                                read_sims.add(sim_id)
                                read_count[step] += 1
                                if train_ai.logger:
                                    train_ai.logger.info(f"Read data: input_{sim_id}_{rank}_{step//update_frequency}")
                    except Exception as e:
                        if train_ai.logger:
                            train_ai.logger.error(f"Error reading data for sim {sim_id}: {e}")
                        read_sims.add(sim_id)
                time.sleep(0.1)  # Avoid busy waiting
            dt_time_total = time.time() - tstart
        else:
            dt_time_total = 0.0
            dt_time = 0.0

        iteration_time[step] = train_time
        data_read_time[step] = dt_time

        if train_ai.logger:
            train_ai.logger.info(f"tstep time: {train_time}, dt time: {dt_time}")
        if init_MPI:
            comm.Barrier()

    if server_info["config"]["type"] != "dragon" and comm is not None:
        mask = iteration_time > 0
        mean, std = compute_mean_std(comm, iteration_time, mask.astype(np.int32))
        if train_ai.logger:
            train_ai.logger.info(f"Mean iteration time (s): {mean}, Std: {std}")
        

        mean, std = compute_mean_std(comm, data_read_time, read_count)
        if train_ai.logger:
            train_ai.logger.info(f"Mean read time (s): {mean}, Std: {std}")

        #compute throughput
        mask = read_count > 0
        throughput = np.zeros_like(data_read_time)
        throughput[mask] = read_count[mask]*np.prod(data.shape)*1e-6*data.dtype.itemsize/data_read_time[mask]
        mean, std = compute_mean_std(comm, throughput, read_count)
        if train_ai.logger:
            train_ai.logger.info(f"Mean read throughput (MB/s): {mean}, Std: {std}")
    elif server_info["config"]["type"] == "dragon":
        ddict = train_ai.dragon_dict
        mean, std = compute_mean_std_dragon(ddict, iteration_time, (iteration_time > 0).astype(np.int32), rank, "trainIterationTime", size)
        if train_ai.logger:
            train_ai.logger.info(f"Mean iteration time (s): {mean}, Std: {std}")
        mean, std = compute_mean_std_dragon(ddict, data_read_time, read_count, rank, "dataReadTime", size)
        if train_ai.logger:
            train_ai.logger.info(f"Mean read time (s): {mean}, Std: {std}")

        #compute throughput
        mask = read_count > 0
        throughput = np.zeros_like(data_read_time)
        throughput[mask] = read_count[mask]*np.prod(data.shape)*1e-6*data.dtype.itemsize/data_read_time[mask]
        mean, std = compute_mean_std_dragon(ddict, throughput, read_count, rank, "readThroughput", size)
        if train_ai.logger:
            train_ai.logger.info(f"Mean read throughput (MB/s): {mean}, Std: {std}")

    train_ai.flush_logger()
    # if rank == 0:
    #     train_ai.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--config", type=str, required=True, help="help")
    parser.add_argument("--server_info", type=str, default=None, help="Server info for the AI")
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), args.config), "r") as f:
        config = json.load(f)
    main(config, 
        server_info=args.server_info)
