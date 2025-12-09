from typing import Dict
from SimAIBench import DataStore
import random
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def infer_main(infer_serverinfo: Dict, top_candidate_serverinfo: Dict):
    import time

    infer_ds = DataStore(name="infer",server_info=infer_serverinfo)
    model_ds = DataStore(name="top_candidates",server_info=top_candidate_serverinfo)

    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Inference process started with rank {rank} out of {size}", flush=True)
    for iter in range(10):
        if model_ds.poll_staged_data(f"new_model_{iter}"):
            new_model = model_ds.stage_read(f"new_model_{iter}")
            print(f"Updated model: {new_model}")
        random_ints = [random.randint(0, 999) for _ in range(100)]
        print(f"Infer iteration {iter}")
        for idx,smile_num in enumerate(random_ints):
            ##read the inference data
            smile = infer_ds.stage_read(f"smile_{smile_num}")
            time.sleep(0.05)
            if rank==0:
                infer_ds.stage_write(f"infer_smile_{idx}_{iter}",(random.random(),smile))