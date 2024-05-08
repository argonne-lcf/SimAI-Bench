#####
##### This script contains training loops, validation loops, and testing loops
##### used during online training from simulation data to be called from the 
##### training driver to assist in learning and evaluation model performance.
#####
import sys
import io
from typing import List
from time import perf_counter
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.cuda.amp import autocast, GradScaler
    from torch.xpu.amp import autocast, GradScaler
except:
    pass

from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import metric_average
from .offline_train import train, validate, test
from .datasets import RankDataset, RankStepDataset

# Generate training and validation data loaders for DB interaction
def setup_online_dataloaders(cfg, comm, dataset, batch_size, split: List[float]):
    # Split tensors for trianing and validation
    # random generator is needed to ensure same split across ranks
    generator = torch.Generator().manual_seed(12345)
    train_dataset, val_dataset = random_split(dataset, split, generator=generator)
 
    if (cfg.online.smartredis.db_launch=="colocated"):
        replicas = cfg.ppn*cfg.ppd
        rank_arg = comm.rankl
    else:
        replicas = comm.size
        rank_arg = comm.rank
    if cfg.model=="gnn":
        train_sampler = None
        val_sampler = None
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=replicas, 
                                           rank=rank_arg, drop_last=True, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=replicas, 
                                         rank=rank_arg, drop_last=False)
    train_tensor_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     sampler=train_sampler)
    val_tensor_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   sampler=val_sampler)

    return train_tensor_loader, train_sampler, val_tensor_loader

### Main online training loop driver
def onlineTrainLoop(cfg, comm, client, t_data, model, logger):
    # Setup and variable initialization
    istep = -1 # initialize the simulation step number
    iepoch = 0 # epoch number

    # Set precision of model
    if (cfg.precision == "fp32" or cfg.precision == "tf32"): model.float()
    elif (cfg.precision == "fp64"): model.double()
    elif (cfg.precision == "fp16"): model.half()
    elif (cfg.precision == "bf16"): model.bfloat16()
    if (cfg.mixed_precision):
        scaler = GradScaler(enabled=True)
        if (cfg.device == "cuda"):
            mixed_dtype = torch.float16
        elif (cfg.device == "xpu"):
            mixed_dtype = torch.bfloat16
    else:
        scaler = None
        mixed_dtype = None
 
    # Wrap model with DDP
    model = DDP(model)

    # Initialize optimizer
    if (cfg.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate*comm.size)
    elif (cfg.optimizer == "RAdam"):
        optimizer = optim.RAdam(model.parameters(), lr=cfg.learning_rate*comm.size)
    else:
        logger.error(f"Optimizer {cfg.optimizer} not implemented at the moment")
    if (cfg.scheduler == "Plateau"):
        if (comm.rank==0): logger.info("Applying plateau scheduler\n")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

    # Create training and validation Datasets
    num_db_tensors = client.num_local_tensors
    num_val_tensors = int(num_db_tensors*cfg.validation_split)
    num_train_tensors = num_db_tensors - num_val_tensors
    tensor_split = [1-cfg.validation_split, cfg.validation_split]
    if (num_val_tensors==0 and cfg.validation_split>0):
        if (comm.rank==0): logger.warning("Insufficient number of tensors for validation -- skipping it")
    if client.dataOverWr:
        key_dataset = RankDataset(num_db_tensors,client.head_rank,cfg.model,comm.rank)

    # While loop that checks when training data is available on database
    if (comm.rank == 0):
        logger.info("Waiting for training data to be populated in DB ...")
    while True:
        if client.key_exists("step"):
            step = client.get_value('step')
            break
    if (comm.rank == 0):
        logger.info("Found data, starting training loop\n")

    # Start training loop
    step_list = []
    tic_loop = perf_counter()
    while True:
        # Check to see if simulation says time to quit, if so break loop
        if client.key_exists("sim-run"):
            sim_run = client.get_value('sim-run')
            if (sim_run < 0.5):
                if (comm.rank == 0):
                    logger.info("Simulation says time to quit training ... \n")
                break

        # Get time step number from database
        step = client.get_value('step')

        # If step number mismatch, create new data loaders and update
        if (istep != step): 
            istep = step
            step_list.append(istep)
            update = True
            if (comm.rank == 0):
                logger.info("New training data was sent to the DB ...")
                logger.info(f"Working with time step {istep} \n")
            if not client.dataOverWr:
                client.tensor_batch =  max(1,int(num_db_tensors*len(step_list)/(cfg.ppn*cfg.ppd)))
                if cfg.model=="gnn": 
                    num_db_tensors = 1
                    client.tensor_batch =  len(step_list)
                key_dataset = RankStepDataset(num_db_tensors, step_list, client.head_rank, cfg.model, comm.rank)
            train_tensor_loader, \
                train_sampler, \
                val_tensor_loader = setup_online_dataloaders(cfg, comm, key_dataset, client.tensor_batch, 
                                                             tensor_split)
        else:
            update = False
        
        # Print epoch number
        if (comm.rank == 0):
            logger.info(f"Epoch {iepoch+1}")
            logger.info("-------------------------------")
        
        # Perform training step
        if train_sampler is not None: train_sampler.set_epoch(iepoch)
        running_loss = 0.
        tic_t = perf_counter()
        for _, tensor_keys in enumerate(train_tensor_loader):
            if (cfg.online.global_shuffling or update):
                train_loader = model.module.online_dataloader(cfg, client, comm,
                                                              tensor_keys, logger,
                                                              shuffle=True)
            loss, t_data = train(comm, model, 
                                 train_loader, optimizer, 
                                 scaler, mixed_dtype, iepoch, 
                                 t_data, cfg, logger)
            
            running_loss += loss

        local_loss = running_loss / len(train_tensor_loader)
        global_loss = metric_average(comm, local_loss)
        toc_t = perf_counter()
        if (iepoch>0):
            nTrain = client.tensor_batch * client.npts
            t_data.t_train = t_data.t_train + (toc_t - tic_t)
            t_data.tp_train = t_data.tp_train + nTrain/(toc_t - tic_t)
            t_data.i_train = t_data.i_train + 1
        if comm.rank == 0: 
            logger.info(f"Training set: | Epoch: {iepoch+1} | Average loss: {global_loss:>8e} \n")

        # Perform validation step
        if (num_val_tensors>0):
            running_loss = 0.
            running_acc = 0.
            tic_v = perf_counter()
            for _, tensor_keys in enumerate(val_tensor_loader):
                if (cfg.online.global_shuffling or update):
                    val_loader = model.module.online_dataloader(cfg, client, comm, 
                                                                tensor_keys, logger)
                acc, loss, _ = validate(comm, model, val_loader, 
                                        mixed_dtype, iepoch, cfg,
                                        logger)
                running_loss += loss
                running_acc += acc
            val_loss = running_loss / len(val_tensor_loader)
            global_val_loss = metric_average(comm, val_loss)
            val_acc = running_acc / len(val_tensor_loader)
            global_val_acc = metric_average(comm, val_acc)
            toc_v = perf_counter()
            if (iepoch>0):
                nVal = len(tensor_keys) * client.npts
                t_data.t_val = t_data.t_val + (toc_v - tic_v)
                t_data.tp_val = t_data.tp_val + nVal/(toc_v - tic_v)
                t_data.i_val = t_data.i_val + 1
            if comm.rank == 0: 
                logger.info(f"Validation set: | Epoch: {iepoch+1} | Average accuracy: {global_val_acc:>8e} | Average Loss: {global_val_loss:>8e}") 

        # Share model checkpoint
        if ((iepoch+1)%cfg.online.checkpoints==0):
            if (comm.rankl==0):
                jit_model = model.module.script_model()
                #model_arr = np.array([model_bytes])
                #client.client.put_tensor(cfg.model,model_arr)
                if (cfg.model=="gnn"):
                    torch.jit.save(jit_model, f"/tmp/{cfg.model}.pt")
                else:
                    buffer = io.BytesIO()
                    torch.jit.save(jit_model, buffer)
                    model_bytes = buffer.getvalue()
                    if client.model_exists(cfg.model):
                        client.delete_model(cfg.model)
                    client.put_model(cfg.model, model_bytes,
                                     device=cfg.online.smartredis.inference_device)
            if (comm.rank==0):
                logger.info("Shared model checkpoint\n")

        iepoch = iepoch + 1 

    toc_loop = perf_counter()
    t_data.t_tot = toc_loop - tic_loop


    # Sync with simulation
    if comm.rankl==0:
        client.put_value('train-run', 0)
 
    sample_data = next(iter(train_loader)) 
    return model, sample_data



