
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from SimAIBench.training import AI
import time
import argparse
import logging as logging_
import json
import socket
import os
import pyitt
from typing import Union


def main(ai_config:str,
         server_info:Union[str, dict]=None,
         is_colocated:bool=False,
         init_MPI=True,
         rank:int=0,
         size:int=1,
         init_time:float=29.0):
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

    infer_ai = AI(f"infer_ai_{rank}", server_info=server_info, num_hidden_layers=1, num_epochs=1, 
                    device=device,logging=rank==0,log_level=logging_.INFO,is_colocated=is_colocated)
    if init_MPI:
        comm.Barrier()
    warmup_time = time.time()
    infer_ai.infer(run_time=run_time)
    warmup_time = time.time() - warmup_time
    if warmup_time < init_time:
        time.sleep(init_time-warmup_time)
    last_update = 0
    for i in range(nsteps):
        tic = time.time()
        if os.getenv("PROFILE_WORKFLOW",None) is not None:
            with pyitt.task(f"inference_step_{i}",domain="training"):
                elap_time = infer_ai.infer(run_time=run_time)
        else:
            elap_time = infer_ai.infer(run_time=run_time)
        train_time = time.time() - tic
        if infer_ai.logger:
            infer_ai.logger.info(f"tstep time: {train_time}, dt time:0.0")
        if init_MPI:
            comm.Barrier()
    
    infer_ai.flush_logger()
    infer_ai.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--config", type=str, required=True, help="help")
    parser.add_argument("--server_info", type=str, default=None, help="Server info for the AI")
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), args.config), "r") as f:
        config = json.load(f)
    main(config, 
        server_info=args.server_info)
