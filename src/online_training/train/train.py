# Import general libraries
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from time import perf_counter
import random
import datetime
import logging
import socket

# Import MPI before torch to avoid error
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# Import ML libraries
import torch
import torch.distributed as dist

# Import online_training modules
from online_training.train.online_train import onlineTrainLoop
from online_training.train.offline_train import offlineTrainLoop
from online_training.train import models
from online_training.train.time_prof import timeStats
from online_training.train import utils
#import online_training.backends as client_backends
try:
    from online_training.backends.smartredis import SmartRedis_Train_Client
except:
    pass
try:
    from online_training.backends.dragon import Dragon_Train_Client
except:
    pass


## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="train_config")
def main(cfg: DictConfig):
    
    # Import and init MPI
    comm = utils.MPI_COMM()
    comm.init(cfg)

    # Set up logging
    log_level = getattr(logging, cfg.logging.upper())
    logger = logging.getLogger(f'[{comm.rank}]')                                
    logger.setLevel(log_level)
    date = datetime.datetime.now().strftime('%d.%m.%y_%H.%M') if comm.rank==0 else None
    date = comm.comm.bcast(date, root=0)
    comm.comm.Barrier()
    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(message)s')
    mh = utils.MPIFileHandler(f"train_{date}.log", comm=comm.comm)                       
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    #fh = logging.FileHandler(f'{os.getcwd()}/train_{date}.log')
    #fh.setFormatter(formatter)
    #if comm.rank==0: logger.addHandler(fh)
    
    logger.debug(f"Hello from MPI rank {comm.rank}/{comm.size} and local rank {comm.rankl}")

    # Intel imports
    try:
        import intel_extension_for_pytorch
    except ModuleNotFoundError as err:
        if comm.rank==0: logger.warning(err)
    try:
        import oneccl_bindings_for_pytorch
    except ModuleNotFoundError as err:
        if comm.rank==0: logger.warning(err)

    # Initialize Torch Distributed
    os.environ['RANK'] = str(comm.rank)
    os.environ['WORLD_SIZE'] = str(comm.size)
    master_addr = socket.gethostname() if comm.rank == 0 else None
    master_addr = comm.comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (cfg.device=='cpu'): backend = 'gloo'
    elif (cfg.device=='cuda' and cfg.ppd==1): backend = 'nccl'
    elif (cfg.device=='cuda' and cfg.ppd>1): backend = 'gloo'
    elif (cfg.device=='xpu'): backend = 'ccl'
    dist.init_process_group(backend,
                            rank=int(comm.rank),
                            world_size=int(comm.size),
                            init_method='env://',
                            timeout=datetime.timedelta(seconds=120))

    # Set all seeds if need reproducibility
    if cfg.reproducibility:
        random_seed = 123456789
        random.seed(a=random_seed)
        np.random.seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.manual_seed(random_seed)
        if (cfg.device=='cuda' and torch.cuda.is_available()):
            torch.cuda.manual_seed_all(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True) # only pick deterministic algorithms
        elif (cfg.device=='xpu' and torch.xpu.is_available()):
            torch.xpu.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True) # only pick deterministic algorithms
    else:
        rng = np.random.default_rng()

    # Instantiate performance data class
    t_data = timeStats()

    # Initialize client if performing online training
    client = None
    if cfg.online.backend:
        if cfg.online.backend=='smartredis':
            client = SmartRedis_Train_Client(cfg, comm.rank, comm.size)
        elif cfg.online.backend=='dragon':
            client = Dragon_Train_Client(cfg, comm.rank, comm.size)
        client.init()
        comm.comm.Barrier()
        if comm.rank == 0:
            logger.info(f"Initialized all {cfg.online.backend} clients\n")
            logger.info("Getting size info from simulation ...")
        client.setup_problem()
        if comm.rank == 0:
            logger.info(f"Samples per simulation tensor: {client.npts}")
            logger.info(f"Model input features: {client.ndIn}")
            logger.info(f"Model output targets: {client.ndOut}")
            logger.info(f"Total tensors in all DB: {client.num_tot_tensors}")
            logger.info(f"Tensors in local DB: {client.num_local_tensors}")
            logger.info(f"Simulation tensors per batch: {client.tensor_batch}")
            logger.info(f"Overwriting simulaiton tensors: {client.dataOverWr}\n")

    # Instantiate the model and get the training data
    model, data = models.load_model(cfg, comm, client, rng)
    n_params = utils.count_weights(model)
    if (comm.rank == 0):
        logger.info(f"Loaded {cfg.model} model with {n_params} trainable parameters \n")
    
    # Set device to run on and offload model
    device = torch.device(cfg.device)
    torch.set_num_threads(1)
    if (cfg.device == 'cuda'):
        if torch.cuda.is_available():
            cuda_id = comm.rankl//cfg.ppd if torch.cuda.device_count()>1 else 0
            assert cuda_id>=0 and cuda_id<torch.cuda.device_count(), \
                   f"Assert failed: cuda_id={cuda_id} and {torch.cuda.device_count()} available devices"
            torch.cuda.set_device(cuda_id)
        else:
            logger.warning(f"[{comm.rank}]: no cuda devices available, cuda.device_count={torch.cuda.device_count()}")
    elif (cfg.device=='xpu'):
        if torch.xpu.is_available():
            xpu_id = comm.rankl//cfg.ppd if torch.xpu.device_count()>1 else 0
            assert xpu_id>=0 and xpu_id<torch.xpu.device_count(), \
                   f"Assert failed: xpu_id={xpu_id} and {torch.xpu.device_count()} available devices"
            torch.xpu.set_device(xpu_id)
        else:
            logger.warning(f"[{comm.rank}]: no XPU devices available, xpu.device_count={torch.xpu.device_count()}")
    if (cfg.device != 'cpu'):
        model.to(device)
    if (comm.rank == 0):
        logger.info(f"Running on device: {cfg.device} \n")

    # Train model
    if cfg.online.backend:
        model, sample_data = onlineTrainLoop(cfg, comm, client, t_data, model, logger)
    else:
        model, sample_data = offlineTrainLoop(cfg, comm, t_data, model, data)

    # Save model to file before exiting
    model = model.module
    dist.destroy_process_group()
    if (comm.rank == 0):
        model.eval()
        model.save_checkpoint(cfg.name, sample_data)
        logger.info("Saved model to disk\n")

    # Collect timing statistics for training
    if (t_data.i_train>0):
        if (comm.rank==0):
            logger.info("Timing data:")
        t_data.printTimeData(comm, logger)

    # Accumulate timing data for client and print summary
    client.collect_stats(comm)
    if comm.rank==0:
        logger.info("Summary of client timing data:")
        client.print_stats(logger)

    # Print FOM
    train_array_sz = client.train_array_sz / 1024 / 1024 / 1024
    if comm.rank==0:
        logger.info("FOM:")
        utils.print_fom(logger, t_data.t_tot, train_array_sz, client.time_stats)

    if cfg.online.backend:
        client.destroy()
    mh.close()
    comm.finalize()


## Run main
if __name__ == "__main__":
    main()
