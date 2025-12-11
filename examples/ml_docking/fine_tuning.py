from typing import Dict
from SimAIBench import DataStore
from SimAIBench.utils import create_logger
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def finetune_main(training_serverinfo: Dict, top_candiate_serverinfo: Dict):
    import time
    #read the simulation/training data
    training_ds = DataStore(name="training",server_info=training_serverinfo)
    model_ds = DataStore(name="model",server_info=top_candiate_serverinfo)

    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = create_logger("finetuning")
    if rank==0: logger.info(f"Finetune process started with rank {rank} out of {size}")
    if rank==0: logger.info(f"Connected to training DataStore: {training_ds.name}")
    if rank==0: logger.info(f"Connected to model DataStore: {model_ds.name}")

    for iter in range(10):
        if rank==0: logger.info(f"Rank {rank}: Training iteration {iter}/10 started")
        smile_strings = []
        y = []
        ref = []
        for idx in range(100):
            key = f"infer_smile_{idx}_{iter}"
            while not training_ds.poll_staged_data(key):
                time.sleep(0.5)
            value = training_ds.stage_read(key,timeout=300)
            smile_strings.append(value[0])
            y.append(value[1])
            ref.append(value[2])
        if rank==0: logger.info(f"Rank {rank}: Collected {len(smile_strings)} samples for iteration {iter}")
        time.sleep(5.0) ##do some training here
        if rank==0: logger.info(f"Rank {rank}: Training completed for iteration {iter}")
        if rank==0:
            model_ds.stage_write(f"new_model_{iter}",f"I am new model number {iter}")
            logger.info(f"Rank 0: Published new model for iteration {iter}")
        
        comm.Barrier()
    
    if rank==0: logger.info(f"Rank {rank}: Finetuning completed all iterations")
    MPI.Finalize()