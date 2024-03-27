#####
##### This script contains training loops, validation loops, and testing loops
##### used during online training from simulation data to be called from the 
##### training driver to assist in learning and evaluation model performance.
#####
import sys
from datetime import datetime
from time import sleep,perf_counter
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
try:
    import horovod.torch as hvd
except:
    pass

from utils import metric_average
from offline_train import train, validate, test
from datasets import MiniBatchDataset, KeyDataset

# Generate training and validation data loaders for DB interaction
def setup_online_dataloaders(cfg, comm, dataset, batch_size, split):
    # Split tensors for trianing and validation
    # random generator is needed to ensure same split across ranks
    generator = torch.Generator().manual_seed(12345)
    train_dataset, val_dataset = random_split(dataset, split, generator=generator)
 
    if (cfg.online.db_launch=="colocated"):
        replicas = cfg.ppn*cfg.ppd
        rank_arg = comm.rankl
    else:
        replicas = comm.size
        rank_arg = comm.rank
    train_sampler = DistributedSampler(train_dataset, num_replicas=replicas, 
                                       rank=rank_arg, drop_last=True, shuffle=True)
    train_tensor_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     sampler=train_sampler)
    val_sampler = DistributedSampler(val_dataset, num_replicas=replicas, 
                                     rank=rank_arg, drop_last=False)
    val_tensor_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   sampler=val_sampler)

    return train_tensor_loader, train_sampler, val_tensor_loader

### Main online training loop driver
def onlineTrainLoop(cfg, comm, client, t_data, model):
    # Setup and variable initialization
    istep = -1 # initialize the simulation step number
    iepoch = 0 # epoch number
    rerun_check = 1 # 0 means quit training

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
 
    # Initializa DDP model
    if (cfg.distributed=='ddp'):
        model = DDP(model,broadcast_buffers=False,gradient_as_bucket_view=True)

    # Initialize optimizer
    if (cfg.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate*comm.size)
    elif (cfg.optimizer == "RAdam"):
        optimizer = optim.RAdam(model.parameters(), lr=cfg.learning_rate*comm.size)
    else:
        print("ERROR: optimizer implemented at the moment")
    if (cfg.scheduler == "Plateau"):
        if (comm.rank==0): print("Applying plateau scheduler\n")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    # Broadcast state if using Horovod
    if (cfg.distributed=='horovod'):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             op=hvd.mpi_ops.Sum,
                                             num_groups=1)

    # Create training and validation Datasets
    tot_db_tensors = client.num_db_tensors*client.nfilters
    sim_rank_list = np.arange(0,tot_db_tensors,dtype=int)
    num_val_tensors = int(tot_db_tensors*cfg.validation_split)
    num_train_tensors = client.num_db_tensors*client.nfilters - num_val_tensors
    tensor_split = [num_train_tensors, num_val_tensors]
    if (num_val_tensors==0 and cfg.validation_split>0):
        if (comm.rank==0): print("Insufficient number of tensors for validation -- skipping it")
    #if (cfg.model=="sgs" and client.nfilters>1):
    #    dataset = KeyMFDataset(sim_rank_list,client.num_db_tensors,
    #                            client.head_rank,client.filters)
    #else:
    key_dataset = KeyDataset(sim_rank_list,client.head_rank,istep,client.dataOverWr)

    # While loop that checks when training data is available on database
    if (comm.rank == 0):
        print("\nWaiting for training data to be populated in DB ...")
        sys.stdout.flush()
    while True:
        if (client.client.poll_tensor("step",0,1)):
            rtime = perf_counter()
            tmp = client.client.get_tensor('step')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1 
            break
    if (comm.rank == 0):
        print("Found data, starting training loop\n")
        sys.stdout.flush()

    # Start training loop
    while True:
        tic_l = perf_counter()
        # Check to see if simulation says time to quit, if so break loop
        if (client.client.poll_tensor("check-run",0,1)):
            rtime = perf_counter()
            tmp = client.client.get_tensor('check-run')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1
            if (tmp[0] < 0.5):
                if (comm.rank == 0):
                    print("Simulation says time to quit training ... \n", flush=True)
                iTest = False
                rerun_check = 0
                break

        # Get time step number from database
        rtime = perf_counter()
        tmp = client.client.get_tensor('step')
        rtime = perf_counter() - rtime
        t_data.t_meta = t_data.t_meta + rtime
        t_data.i_meta = t_data.i_meta + 1

        # If step number mismatch, create new data loaders and update
        if (istep != tmp[0]): 
            istep = tmp[0]
            update = True
            if (comm.rank == 0):
                print("\nNew training data was sent to the DB ...")
                print(f"Working with time step {istep} \n", flush=True)
            train_tensor_loader, \
                train_sampler, \
                val_tensor_loader = setup_online_dataloaders(cfg, comm, key_dataset, client.tensor_batch, 
                                                             tensor_split)
        else:
            update = False
        
        # Print epoch number
        if (comm.rank == 0):
            print(f"\n Epoch {iepoch+1} of {cfg.epochs}")
            print("-------------------------------", flush=True)
        
        # Perform training step
        train_sampler.set_epoch(iepoch)
        tic_t = perf_counter()
        running_loss = 0.
        for _, tensor_keys in enumerate(train_tensor_loader):
            if (cfg.online.global_shuffling or update):
                if (cfg.distributed=='horovod'):
                    train_loader, rtime = model.online_dataloader(cfg, client, comm, 
                                                                  tensor_keys,
                                                                  shuffle=True)
                elif (cfg.distributed=='ddp'):
                    train_loader, rtime = model.module.online_dataloader(cfg, client, comm,
                                                                         tensor_keys,
                                                                         shuffle=True)
                if (iepoch>0):
                    t_data.t_getBatch = t_data.t_getBatch + rtime
                    t_data.i_getBatch = t_data.i_getBatch + 1
                    fact = float(1.0/t_data.i_getBatch)
                    t_data.t_AveGetBatch = fact*rtime + (1.0-fact)*t_data.t_AveGetBatch
            
            running_loss, t_data = train(comm, model, 
                                 train_loader, optimizer, 
                                 scaler, mixed_dtype, iepoch, 
                                 t_data, cfg)
            running_loss += running_loss

        loss = running_loss / len(train_tensor_loader)
        global_loss = metric_average(comm, loss)
        toc_t = perf_counter()
        if (iepoch>0):
            t_data.t_train = t_data.t_train + (toc_t - tic_t)
            t_data.i_train = t_data.i_train + 1
        if comm.rank == 0: 
            print(f"Training set: | Epoch: {iepoch+1} | Average loss: {global_loss:>8e} \n", flush=True)

        # Perform validation step
        if (num_val_tensors>0):
            running_loss = 0.
            running_acc = 0.
            tic_v = perf_counter()
            for _, tensor_keys in enumerate(val_tensor_loader):
                if (cfg.online.global_shuffling or update):
                    if (cfg.distributed=='horovod'):
                        val_loader, rtime = model.online_dataloader(cfg, client, comm, tensor_keys)
                    elif (cfg.distributed=='ddp'):
                        val_loader, rtime = model.module.online_dataloader(cfg, client, comm, tensor_keys)
                    if (iepoch>0):
                        t_data.t_getBatch_v = t_data.t_getBatch_v + rtime
                        t_data.i_getBatch_v = t_data.i_getBatch_v + 1
                        fact = float(1.0/t_data.i_getBatch_v)
                        t_data.t_AveGetBatch_v = fact*rtime + (1.0-fact)*t_data.t_AveGetBatch_v
                running_acc, running_loss, testData = validate(comm, model, val_loader, 
                                                      mixed_dtype, iepoch, cfg)
                running_loss += running_loss
                running_acc += running_acc
            val_loss = running_loss / len(val_tensor_loader)
            global_val_loss = metric_average(comm, val_loss)
            val_acc = running_acc / len(val_tensor_loader)
            global_val_acc = metric_average(comm, val_acc)
            toc_v = perf_counter()
            if (iepoch>0):
                t_data.t_val = t_data.t_val + (toc_v - tic_v)
                t_data.i_val = t_data.i_val + 1
            if comm.rank == 0: 
                print(f"Validation set: | Epoch: {iepoch+1} | Average accuracy: {global_val_acc:>8e} | Average Loss: {global_val_loss:>8e}")        
        
        # Time entire loop
        toc_l = perf_counter()
        if (iepoch>0):
            t_data.t_tot = t_data.t_tot + (toc_l - tic_l)
        
        # Check if tolerance on loss is satisfied
        if (global_loss <= cfg.tolerance):
            if (comm.rank == 0):
                print("\nConvergence tolerance met. Stopping training loop. \n", flush=True)
            iTest = True
            break
        
        # Check if max number of epochs is reached
        if (iepoch >= cfg.epochs-1):
            if (comm.rank == 0):
                print("\nMax number of epochs reached. Stopping training loop. \n", flush=True)
            iTest = True
            break

        iepoch = iepoch + 1 


    # Perform testing on a new snapshot
    if (iTest):
        if (comm.rank==0):
            print("\nTesting model\n-------------------------------", flush=True)
 
        # Wait for new data to be sent to DB
        while True:
            if (client.client.poll_tensor("step",0,1)):
                tmp = client.client.get_tensor('step')
            
            if (istep != tmp[0]):
                istep = tmp[0]
                if (comm.rank == 0):
                    print(f"Working with time step {istep} \n", flush=True)
                break

        # Create dataset, samples and loader for the test data
        test_dataset = KeyDataset(sim_rank_list,client.head_rank,istep,client.dataOverWr)
        test_tensor_loader, test_sampler, _ = setup_online_dataloaders(cfg, comm, test_dataset, client.tensor_batch, 
                                                                       [len(sim_rank_list), 0])

        # Call testing function
        running_loss = 0.
        running_acc = 0.
        for _, tensor_keys in enumerate(test_tensor_loader):
            if (cfg.distributed=='horovod'):
                test_loader, rtime = model.online_dataloader(cfg, client, comm, tensor_keys)
            elif (cfg.distributed=='ddp'):
                test_loader, rtime = model.module.online_dataloader(cfg, client, comm, tensor_keys)
            running_acc, running_loss, testData = test(comm, model, test_loader, 
                                                       mixed_dtype, cfg)
            running_loss += running_loss
            running_acc += running_acc
        test_loss = running_loss / len(test_tensor_loader)
        global_test_loss = metric_average(comm, test_loss)
        test_acc = running_acc / len(test_tensor_loader)
        global_test_acc = metric_average(comm, test_acc)
        if comm.rank == 0: 
            print(f"Test set: Average accuracy: {global_test_acc} | Average Loss: {global_test_loss:>8e}")        

    # Tell simulation to quit
    if (comm.rankl==0 and rerun_check!=0):
        if (comm.rank==0): 
            print("Telling simulation to quit ... \n")
        arrMLrun = np.zeros(2)
        client.client.put_tensor("check-run",arrMLrun)
 
    return model, testData



