from typing import Dict
from SimAIBench import DataStore
import random
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def sim_main(top_candidate_serverinfo: Dict, training_serverinfo:Dict):
    import time

    top_candidate_ds = DataStore(name="top_candidates",server_info=top_candidate_serverinfo)
    training_ds = DataStore(name="training",server_info=training_serverinfo)

    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Docking simulation process started with rank {rank} out of {size}", flush=True)

    for iter in range(10):
        data = {}
        print(f"Docking simulation: {iter}")
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            while not top_candidate_ds.poll_staged_data(key):
                time.sleep(0.05)
            value = top_candidate_ds.stage_read(key,timeout=300)
            time.sleep(0.05) #run a docking sim
            data[key] = (value[1],value[0],random.random())
            if rank==0:
                training_ds.stage_write(key,data[key])