from typing import Dict
from SimAIBench import DataStore
from SimAIBench.utils import create_logger
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
    logger = create_logger("inference")
    if rank==0: logger.info(f"Inference process started with rank {rank} out of {size}")
    if rank==0: logger.info(f"Connected to DataStores: infer and top_candidates")
    
    for iter in range(10):
        if rank==0: logger.info(f"Rank {rank}: Starting inference iteration {iter}")
        if model_ds.poll_staged_data(f"new_model_{iter}"):
            new_model = model_ds.stage_read(f"new_model_{iter}")
            if rank==0: logger.info(f"Rank {rank}: Updated model at iteration {iter}: {new_model}")
        else:
            if rank==0: logger.info(f"Rank {rank}: No new model available at iteration {iter}")
        
        random_ints = [random.randint(0, 999) for _ in range(100)]
        if rank==0: logger.info(f"Rank {rank}: Processing {len(random_ints)} SMILES for iteration {iter}")
        
        for idx,smile_num in enumerate(random_ints):
            ##read the inference data
            smile = infer_ds.stage_read(f"smile_{smile_num}")
            time.sleep(0.05)
            if rank==0:
                infer_ds.stage_write(f"infer_smile_{idx}_{iter}",(random.random(),smile))
                if idx % 25 == 0:  # Log every 25 processed SMILES
                    if rank==0: logger.info(f"Rank {rank}: Processed {idx+1}/{len(random_ints)} SMILES in iteration {iter}")
        
        comm.Barrier()
        if rank==0:
            logger.info(f"Rank {rank}: Completed inference iteration {iter}, processed {len(random_ints)} SMILES")
    
    if rank==0: logger.info(f"Rank {rank}: Inference process completed")
    MPI.Finalize()