from typing import Dict, Tuple
from SimAIBench import DataStore
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def sorter_main(infer_serverinfo: Dict, top_candidate_serverinfo: Dict):
    import time
    
    infer_ds = DataStore(name="inferred data",server_info=infer_serverinfo)
    top_candidate_ds = DataStore(name="top candidates",server_info=top_candidate_serverinfo)

    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Sorter process started with rank {rank} out of {size}", flush=True)

    for iter in range(10):
        print(f"Sorting iter: {iter}")
        data = {}
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            while not infer_ds.poll_staged_data(key):
                time.sleep(0.5)
            data[key] = infer_ds.stage_read(key,timeout=300)
        time.sleep(5.0) ##sort the smiles based on the docking scores
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            if rank==0:
                top_candidate_ds.stage_write(key,data[key])