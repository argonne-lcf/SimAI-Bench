import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError as e:
    pass    
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
try:
    import oneccl_bindings_for_pytorch
except ModuleNotFoundError as e:
    pass
from SimAIBench.datastore import DataStore
import time
import socket
import sys
import os
import datetime
import math
import gc
import logging as logging_
import numpy as np
from typing import Union

class SimpleFeedForwardNet(nn.Module):
    def __init__(self, 
                 dropout=0.1, 
                 use_batchnorm=True, 
                 num_hidden_layers=0, 
                 input_dim=256, 
                 output_dim=256, 
                 neurons_per_layer=256):
        super(SimpleFeedForwardNet, self).__init__()
        
        self.model = nn.Sequential()
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons_per_layer = neurons_per_layer
        self.input_params = {
                        "dropout":dropout,
                        "use_batchnorm":use_batchnorm,
                        "num_hidden_layers":num_hidden_layers,
                        "input_dim":input_dim,
                        "output_dim":output_dim,
                        "neurons_per_layer":neurons_per_layer
                     }

        self.layers = [nn.Linear(input_dim, neurons_per_layer)]
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(neurons_per_layer))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*self.layers)
    
    def get_input_params(self):
        return self.input_params
    
    def forward(self, x):
        return self.model(x)
        
def setup_dataloader(data_set_size, x_shape, y_shape, batch_size=32, shuffle=True, ddp=False):
    X = torch.randn((data_set_size, *x_shape), dtype=torch.float32)
    y = torch.randn((data_set_size, *y_shape), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    if ddp:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_loss_function(loss_type="mse"):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

def train(model, dataloader, criterion, optimizer, device, num_epochs=10,ddp=False):
    model.train()
    for epoch in range(num_epochs):
        if ddp and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)


class AI(DataStore):
    def __init__(self,
                 name = "AI", 
                 server_info:dict = {"type":"filesystem"},
                model_type="feedforward", 
                dropout=0.1, 
                use_batchnorm=True, 
                num_hidden_layers=1, 
                loss_type="mse", 
                lr=0.001, 
                data_size=1000,
                input_dim=256,
                output_dim=256,
                neurons_per_layer=256, 
                batch_size=32, 
                shuffle=True, 
                num_epochs=10,
                device="cpu",
                ddp=False, 
                comm=None,
                logging=False,
                log_level=logging_.INFO,
                **kwargs):
        super().__init__(name,server_info=server_info,logging=logging,log_level=log_level, is_colocated=kwargs.get("is_colocated", False))
        self.name = name
        self.model_type = model_type
        self.loss_type = loss_type
        self.lr = lr
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        assert device in ["cpu", "cuda", "xpu"], "Device must be 'cpu', 'cuda', or 'xpu'."
        self.device = device
        self.ddp = ddp
        self.local_rank = 0
        self.world_size = 1
        self.master_addr = socket.gethostname()
        if self.ddp:
            assert comm is not None, "Distributed training requires a communicator."
            self.comm = comm
            self.local_rank = comm.Get_rank()
            self.world_size = comm.Get_size()
            if self.local_rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None
            self.master_addr = self.comm.bcast(master_addr, root=0)
        
        sys.stdout.flush()
        self.criterion = None
        self.optimizer = None
        self.dataloader = None

        if self.ddp:
            assert device == "cuda" or device == "xpu" or device == "cpu", "DDP is only supported on CUDA or XPU devices."
            if self.device == 'cuda':
                if torch.cuda.is_available():
                    # if torch.cuda.device_count() > 1:
                    #     raise RuntimeError("SI only supports single device training")
                    torch.cuda.set_device(0)
                else:
                    print("CUDA is not available. Exiting ...")
                    sys.exit(1)
            elif (self.device == 'xpu'):
                if torch.xpu.is_available():
                    # if torch.xpu.device_count() > 1:
                    #     raise RuntimeError("This script only supports single device training")
                    torch.xpu.set_device(0)
                else:
                    print("XPU is not available. Exiting ...")
                    sys.exit(1)
        self.model = self.build_model(
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            num_hidden_layers=num_hidden_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_layer=neurons_per_layer
            )
        self.setup_training()
            
    def build_model(self,             
                    dropout=0.1, 
                    use_batchnorm=True, 
                    num_hidden_layers=0, 
                    input_dim=256, 
                    output_dim=256, 
                    neurons_per_layer=256):
        """Build a neural network model."""
        if self.model_type == "feedforward":
            model = SimpleFeedForwardNet(dropout=dropout, 
                                              use_batchnorm=use_batchnorm, 
                                              num_hidden_layers=num_hidden_layers,
                                              input_dim=input_dim,
                                              output_dim=output_dim,
                                              neurons_per_layer=neurons_per_layer)
            if self.ddp:
                if not torch.distributed.is_initialized():
                    os.environ['MASTER_ADDR'] = self.master_addr
                    os.environ['MASTER_PORT'] = '12355'
                    os.environ['WORLD_SIZE'] = str(self.world_size)
                    os.environ['RANK'] = str(self.local_rank)
                    if self.device == 'cuda':
                        torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=30))
                    elif self.device == 'xpu':
                        torch.distributed.init_process_group(backend='ccl', init_method='env://', timeout=datetime.timedelta(seconds=30))
                    else:
                        torch.distributed.init_process_group(backend='gloo', init_method='env://', timeout=datetime.timedelta(seconds=30))
                model = torch.nn.parallel.DistributedDataParallel(model)
            if self.device != 'cpu':
                model.to(self.device)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elif self.device == 'xpu':
                    torch.xpu.synchronize()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        return model
        
    def setup_training(self):
        """Set up loss function and optimizer for training."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
            
        self.criterion = create_loss_function(self.loss_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        input_dim = self.model.input_dim if not self.ddp else self.model.module.input_dim
        output_dim = self.model.output_dim if not self.ddp else self.model.module.output_dim
        self.dataloader = setup_dataloader(  self.data_size, 
                                        (input_dim,),
                                        (output_dim,), 
                                        self.batch_size, 
                                        self.shuffle, 
                                        self.ddp)
            
    def train(self,run_time:Union[float,np.ndarray]=None,run_count:Union[int,np.ndarray]=None)->float:
        """Train the model using data specifications to build a dataloader."""
        assert run_time is not None or run_count is not None
        assert self.model is not None, "Model not built yet. Call build_model first."
        tic = time.time()
        elapsed_time = time.time() - tic
        if self.criterion is None or self.optimizer is None:
            self.setup_training()
        if run_time is not None:
            if isinstance(run_time, np.ndarray):
                current_runtime = np.random.choice(run_time[0], p=run_time[1])
            else:
                current_runtime = run_time
            rc = 0
            while elapsed_time < current_runtime:
                train(self.model, self.dataloader, self.criterion, self.optimizer, self.device, self.num_epochs, self.ddp)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elif self.device == 'xpu':
                    torch.xpu.synchronize()
                elapsed_time = time.time() - tic
                rc += 1
            if self.logger:
                self.logger.debug(f"Elapsed time {elapsed_time} target time {current_runtime} run count {rc}")
            return elapsed_time, rc
        else:
            if isinstance(run_count, np.ndarray):
                current_runcount = np.random.choice(run_count[0], p=run_count[1])
            else:
                current_runcount = run_count
            for _ in range(current_runcount):
                train(self.model, self.dataloader, self.criterion, self.optimizer, self.device, self.num_epochs, self.ddp)
                if self.device == 'cuda':   
                    torch.cuda.synchronize()
                elif self.device == 'xpu':
                    torch.xpu.synchronize()
            elapsed_time = time.time() - tic
            if self.logger:
                self.logger.debug(f"Elapsed time {elapsed_time} run count {current_runcount}")
            return elapsed_time, current_runcount

    def infer(self,run_time:Union[float,np.ndarray]=None,run_count:Union[int,np.ndarray]=None)->float:
        """Perform inference on inputs."""
        assert run_time is not None or run_count is not None
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        tic = time.time()
        elapsed_time = time.time() - tic
        input_dim = self.model.input_dim if not self.ddp else self.model.module.input_dim
        inputs = torch.randn((self.batch_size, input_dim), dtype=torch.float32)
        inputs = inputs.to(self.device)
        self.model.eval()
        if run_time is not None:
            if isinstance(run_time, np.ndarray):
                current_runtime = np.random.choice(run_time[0], p=run_time[1])
            else:
                current_runtime = run_time
            with torch.no_grad():
                rc = 0
                while elapsed_time < current_runtime:
                    outputs = self.model(inputs)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    elif self.device == 'xpu':
                        torch.xpu.synchronize()
                    elapsed_time = time.time() - tic
                    rc += 1
                if self.logger:
                    self.logger.debug(f"Elapsed time {elapsed_time} target time {current_runtime} run count {rc}")
                return elapsed_time, rc
        else:
            if isinstance(run_count, np.ndarray):
                current_runcount = np.random.choice(run_count[0], p=run_count[1])
            else:
                current_runcount = run_count
            with torch.no_grad():
                for _ in range(current_runcount):
                    outputs = self.model(inputs)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    elif self.device == 'xpu':
                        torch.xpu.synchronize()
                elapsed_time = time.time() - tic
                if self.logger:
                    self.logger.debug(f"Elapsed time {elapsed_time} run count {current_runcount}")
            return elapsed_time, current_runcount



# def set_model_params_from_train_time(self,target_time:float):
    #     nn_params = self.model.get_input_params()
    #     ##do some warmup\
    #     for i in range(10):
    #         self.train()
    #     tic = time.time()
    #     for i in range(100):
    #         self.train()
    #     toc = time.time()
    #     train_time = (toc - tic)/100.0
    #     dn = int(((abs(train_time - target_time)/train_time)*nn_params["neurons_per_layer"])//10)
    #     dn = max(dn,256)
    #     if self.logger:
    #         self.logger.info(f"Setting model parameter from training time.target_time:{target_time},train_time{train_time}")
    #     if train_time > target_time:
    #         while train_time > target_time and nn_params["neurons_per_layer"] > 0:
    #             self.model = self.build_model(**nn_params)
    #             tic = time.time()
    #             self.train()
    #             toc = time.time()
    #             train_time = (toc - tic)
    #             if self.logger:
    #                 self.logger.info(f"Current neurons per layer {nn_params['neurons_per_layer']} {train_time} {target_time} {dn}")
    #             nn_params["neurons_per_layer"] -= dn
    #     else:
    #         while train_time < target_time:
    #             self.model = self.build_model(**nn_params)
    #             tic = time.time()
    #             self.train()
    #             toc = time.time()
    #             train_time = (toc - tic)
    #             if self.logger:
    #                 self.logger.info(f"number of neuorns per layer {nn_params['neurons_per_layer']} {train_time} {target_time} {dn}")
    #             nn_params["neurons_per_layer"] += dn
    #     if self.logger:
    #         self.logger.info("Done tuning training parameters!")
    #     return 

    # def set_model_params_from_infer_time(self,target_time:float):
    #     nn_params = self.model.get_input_params()
    #     ##do some warmup
    #     for i in range(10):
    #         self.infer()
    #     tic = time.time()
    #     for i in range(100):
    #         self.infer()
    #     toc = time.time()
    #     infer_time = (toc - tic)/100.0
    #     dn = int(((abs(infer_time - target_time)/infer_time)*nn_params["neurons_per_layer"])//10)
    #     dn = max(dn,256)
    #     if infer_time > target_time:
    #         while infer_time > target_time and nn_params["neurons_per_layer"] > 0:
    #             self.model = self.build_model(**nn_params)
    #             tic = time.time()
    #             self.infer()
    #             toc = time.time()
    #             infer_time = (toc - tic)
    #             if self.logger:
    #                 self.logger.info(f"number of neuorns per layer {nn_params['neurons_per_layer']} {infer_time} {target_time} {dn}")
    #             nn_params["neurons_per_layer"] -= dn

    #     else:
    #         while infer_time < target_time:
    #             self.model = self.build_model(**nn_params)
    #             tic = time.time()
    #             self.infer()
    #             toc = time.time()
    #             infer_time = (toc - tic)
    #             if self.logger:
    #                 self.logger.info(f"number of neuorns per layer {nn_params['neurons_per_layer']} {infer_time} {target_time}")
    #             nn_params["neurons_per_layer"] += dn


    #     if self.logger:
    #         self.logger.info("Done tuning training parameters!")
    #     return 
    