#####
##### This script contains training loops, validation loops, and testing loops
##### used during online training from simulation data to be called from the 
##### training driver to assist in learning and evaluation model performance.
#####
import sys
from time import perf_counter

import torch
import torch.optim as optim
try:
    from torch.cuda.amp import autocast, GradScaler
    from torch.xpu.amp import autocast, GradScaler
except:
    pass

from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import metric_average

### Train the model
def train(comm, model, train_loader, optimizer, scaler, mixed_dtype, 
          epoch, t_data, cfg):
    model.train()
    num_batches = len(train_loader)
    running_loss = torch.tensor([0.0],device=torch.device(cfg.device))

    # Loop over mini-batches
    for batch_idx, data in enumerate(train_loader):
            # Offload batch data
            if (cfg.device != 'cpu'):
               data = data.to(cfg.device)

            # Perform forward and backward passes
            rtime = perf_counter()
            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision, dtype=mixed_dtype):
                loss = model.module.training_step(data)
            if (cfg.mixed_precision):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            rtime = perf_counter() - rtime
            if (epoch>0):
                t_data.t_compMiniBatch = t_data.t_compMiniBatch + rtime
                t_data.i_compMiniBatch = t_data.i_compMiniBatch + 1 
                fact = float(1.0/t_data.i_compMiniBatch)
                t_data.t_AveCompMiniBatch = fact*rtime + (1.0-fact)*t_data.t_AveCompMiniBatch            

            # Update running loss
            running_loss += loss

            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank==0 and (batch_idx)%10==0):
                print(f'{comm.rank}: Train Epoch: {epoch+1} | ' + \
                      f'[{batch_idx+1}/{num_batches}] | ' + \
                      f'Loss: {loss.item():>8e}', flush=True)

    running_loss = running_loss.item() / num_batches
    return running_loss, t_data

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Validate the model
def validate(comm, model, val_loader, mixed_dtype, epoch, cfg):

    model.eval()
    num_batches = len(val_loader)
    running_acc = torch.tensor([0.0],device=torch.device(cfg.device))
    running_loss = torch.tensor([0.0],device=torch.device(cfg.device))

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Offload batch data
            if (cfg.device != 'cpu'):
                data = data.to(cfg.device)

            # Perform forward pass
            with autocast(enabled=cfg.mixed_precision, dtype=mixed_dtype):
                acc, loss = model.module.validation_step(data)
            running_acc += acc
            running_loss += loss
                
            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank==0 and (batch_idx)%50==0):
                print(f'{comm.rank}: Validation Epoch: {epoch+1} | ' + \
                        f'[{batch_idx+1}/{num_batches}] | ' + \
                        f'Accuracy: {acc.item():>8e} | Loss {loss.item():>8e}')
                sys.stdout.flush()

    running_acc = running_acc.item() / num_batches
    running_loss = running_loss.item() / num_batches

    if (cfg.model=='mlp'):
        valData = data[:,:cfg.mlp.inputs]
    else:
        valData = data

    return running_acc, running_loss, valData

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Test the model
def test(comm, model, test_loader, mixed_dtype, cfg):

    model.eval()
    num_batches = len(test_loader)
    running_acc = torch.tensor([0.0],device=torch.device(cfg.device))
    running_loss = torch.tensor([0.0],device=torch.device(cfg.device))

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Offload batch data
            if (cfg.device != 'cpu'):
                data = data.to(cfg.device)

            # Perform forward pass
            with autocast(enabled=cfg.mixed_precision, dtype=mixed_dtype):
                acc, loss = model.module.test_step(data, return_loss=True)
            running_acc += acc
            running_loss += loss
                
            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank==0 and (batch_idx)%50==0):
                print(f'{comm.rank}: Testing | ' + \
                        f'[{batch_idx+1}/{num_batches}] | ' + \
                        f'Accuracy: {acc.cpu().tolist()} | Loss {loss.item():>8e}', flush=True)

    running_acc = running_acc.cpu().numpy() / num_batches
    running_loss = running_loss.item() / num_batches

    if (cfg.model=='mlp'):
        testData = data[:,:cfg.mlp.inputs]
    else:
        testData = data

    return running_acc, running_loss, testData

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Main online training loop driver
def offlineTrainLoop(cfg, comm, t_data, model, data):
    """
    Set up and execute the loop over epochs for offline learning
    """
    # Set precision of model and data
    if (cfg.precision == "fp32" or cfg.precision == "tf32"):
        model.float()
        data = torch.tensor(data, dtype=torch.float32)
    elif (cfg.precision == "fp64"):
        model.double()
        data = torch.tensor(data, dtype=torch.float64)
    elif (cfg.precision == "fp16"):
        model.half()
        data = torch.tensor(data, dtype=torch.float16)
    elif (cfg.precision == "bf16"):
        model.bfloat16()
        data = torch.tensor(data, dtype=torch.bfloat16)
    if (cfg.mixed_precision):
        scaler = GradScaler(enabled=True)
        if (cfg.device == "cuda"):
            mixed_dtype = torch.float16
        elif (cfg.device == "xpu"):
            mixed_dtype = torch.bfloat16
    else:
        scaler = None
        mixed_dtype = None
    
    # Offload entire data (this was actually slower...)
    #if (cfg.device != 'cpu'):
    #    data = data.to(cfg.device)

    # Prepare training and validation data loaders
    loaders = model.setup_dataloaders(data, cfg, comm)
    train_loader = loaders["train"]["loader"]
    train_sampler = loaders["train"]["sampler"]
    nTrain = loaders["train"]["samples"]
    val_loader = loaders["validation"]["loader"]
    nVal = loaders["validation"]["samples"]
    
    # Wrap model with DDP
    model = DDP(model,broadcast_buffers=False,gradient_as_bucket_view=True)

    # Initialize optimizer and scheduler
    if (cfg.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate*comm.size)
    elif (cfg.optimizer == "RAdam"):
        optimizer = optim.RAdam(model.parameters(), lr=cfg.learning_rate*comm.size)
    else:
        print("ERROR: optimizer not implemented at the moment")
    if (cfg.scheduler == "Plateau"):
        if (comm.rank==0): print("Applying plateau scheduler\n")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

    # Loop over epochs
    for epoch in range(cfg.epochs):
        tic_l = perf_counter()
        if (comm.rank == 0):
            print(f"\n Epoch {epoch+1} of {cfg.epochs}")
            print("-------------------------------")
            sys.stdout.flush()

        # Train
        if train_sampler:
            train_sampler.set_epoch(epoch)
        tic_t = perf_counter()
        loss, t_data = train(comm, model, train_loader, 
                             optimizer, scaler, mixed_dtype,
                             epoch, t_data, cfg)
        global_loss = metric_average(comm, loss)
        toc_t = perf_counter()
        if comm.rank == 0: 
            print(f"Training set: | Epoch: {epoch+1} | Average loss: {global_loss:>8e}", flush=True)
        if (epoch>0):
            t_data.t_train = t_data.t_train + (toc_t - tic_t)
            t_data.tp_train = t_data.tp_train + nTrain/(toc_t - tic_t)
            t_data.i_train = t_data.i_train + 1

        # Validate
        if (val_loader is not None):
            tic_v = perf_counter()
            val_acc, val_loss, valData = validate(comm, model, 
                                                  val_loader, 
                                                  mixed_dtype, epoch, cfg)
            global_val_acc = metric_average(comm, val_acc)
            global_val_loss = metric_average(comm, val_loss)
            toc_v = perf_counter()
            if comm.rank == 0:
                print(f"Validation set: | Epoch: {epoch+1} | Average accuracy: {global_val_acc:>8e} | Average Loss: {global_val_loss:>8e}")
            if (epoch>0):
                t_data.t_val = t_data.t_val + (toc_v - tic_v)
                t_data.tp_val = t_data.tp_val + nVal/(toc_v - tic_v)
                t_data.i_val = t_data.i_val + 1
        else:
            global_val_loss = global_loss
            if (cfg.model=='mlp'):
                valData = data[cfg.mini_batch,:cfg.mlp.inputs].to(cfg.device)
            else:
                valData = data.to(cfg.device)

        # Apply scheduler
        if (cfg.scheduler == "Plateau"):
            scheduler.step(global_val_loss)
        
        # Time entire loop
        toc_l = perf_counter()
        if (epoch>0):
            t_data.t_tot = t_data.t_tot + (toc_l - tic_l)

        # Check if tolerance on loss is satisfied
        if (global_val_loss <= cfg.tolerance):
            if (comm.rank == 0):
                print("\nConvergence tolerance met. Stopping training loop. \n")
            break
        
        # Check if max number of epochs is reached
        if (epoch >= cfg.epochs-1):
            if (comm.rank == 0):
                print("\nMax number of epochs reached. Stopping training loop. \n")
            break

    return model, valData
