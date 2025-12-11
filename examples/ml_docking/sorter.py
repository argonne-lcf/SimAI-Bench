from typing import Dict, Tuple
from SimAIBench import DataStore
from SimAIBench.utils import create_logger
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
    logger = create_logger("sorter")
    if rank==0: logger.info(f"Sorter process started with rank {rank} out of {size}")
    if rank==0: logger.info(f"Connected to DataStores: infer_ds and top_candidate_ds")
    
    for iter in range(10):
        if rank==0: logger.info(f"Sorting iter: {iter}")
        data = {}
        if rank==0: logger.info(f"Rank {rank}: Starting to read 100 inferred smiles for iteration {iter}")
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            while not infer_ds.poll_staged_data(key):
                time.sleep(0.5)
            data[key] = infer_ds.stage_read(key,timeout=300)
        if rank==0: logger.info(f"Rank {rank}: Completed reading all 100 smiles for iteration {iter}")
        
        time.sleep(5.0) ##sort the smiles based on the docking scores
        if rank==0: logger.info(f"Rank {rank}: Sorting completed for iteration {iter}")
        
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            if rank==0:
                top_candidate_ds.stage_write(key,data[key])
        
        comm.Barrier()
        if rank==0:
            logger.info(f"Rank {rank}: Done sorting 100 top candidates for iteration {iter}")
    
    if rank==0: logger.info(f"Rank {rank}: Sorting completed all iterations")
    MPI.Finalize()