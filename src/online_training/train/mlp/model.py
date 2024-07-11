################################################
######## MLP ##############################
################################################

from typing import Optional, Tuple
from omegaconf import DictConfig
from time import perf_counter
import numpy as np
import math as m
from numpy import linalg as la
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
try: 
    import vtk
    from vtk.util import numpy_support as VN
except:
    pass

from ..datasets import MiniBatchDataset


class MLP(nn.Module): 
    def __init__(self, inputDim: Optional[int] = 6, outputDim: Optional[int] = 6, 
                 numNeurons: Optional[int] = 20, numLayers: Optional[int] = 1):
        """
        Initialize the MLP model
    
        :param inputDim: optional parameter for the number of input features of the model
        :param outputDim: optional parameter for the number of output targets of the model
        :param numNeurons: optional parameter for the number of neurons in each layer of the model
        :param numLayers: optional parameter for the number of layers of the model
        """
        
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.nLayers = numLayers
        self.min_val = np.zeros((self.ndOut,))
        self.max_val = np.ones((self.ndOut,))
        self.scaler_fit = False

        # Define model network
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(self.ndIn, self.nNeurons)) # input layer
        self.net.append(nn.LeakyReLU(0.3))
        for l in range(self.nLayers-1): # hidden layers
            self.net.append(nn.Linear(self.nNeurons, self.nNeurons))
            self.net.append(nn.LeakyReLU(0.3))
        self.net.append(nn.Linear(self.nNeurons, self.ndOut)) # output layer
        
        # Define the loss function
        self.loss_fn = nn.functional.mse_loss
        # Define the loss function to measure accuracy
        self.acc_fn = nn.functional.mse_loss #comp_corrCoeff

    # Define the method to do a forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        :param x: Input tensor
        :return: output tensor
        """

        for layer in self.net:
            x = layer(x)
        return x

    # Define the methods to do a training, validation and test step
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Perform a training step

        :param batch: a tensor containing the batched inputs and outputs
        :return: loss for the batch
        """

        features = batch[:, :self.ndIn]
        target = batch[:, self.ndIn:]
        output = self.forward(features)
        loss = self.loss_fn(output, target)
        return loss

    def validation_step(self, batch: torch.Tensor) -> Tuple[(torch.Tensor, torch.Tensor)]:
        """
        Perform a validation step

        :param batch: a tensor containing the batched inputs and outputs
        :return: tuple with the accuracy and loss for the batch
        """
        
        features = batch[:, :self.ndIn]
        target = batch[:, self.ndIn:]
        prediction = self.forward(features)
        loss = self.loss_fn(prediction, target)
        acc = self.acc_fn(prediction, target)
        return acc, loss

    def test_step(self, batch: torch.Tensor, return_loss: Optional[bool] = False) -> Tuple[(torch.Tensor, torch.Tensor)]:
        """
        Perform a test step

        :param batch: a tensor containing the batched inputs and outputs
        :param return_loss: whether to compute the loss on the testing data
        :return: tuple with the accuracy and loss for the batch
        """
 
        features = batch[:, :self.ndIn]
        target = batch[:, self.ndIn:]
        prediction = self.forward(features)
        acc = self.acc_fn(prediction, target)
        if return_loss:
            loss = self.loss_fn(prediction, target)
        else:
            loss = torch.Tensor([0.])
        return acc, loss
    
    def create_data(self, cfg: DictConfig, rng) -> np.ndarray:
        """"
        Create synthetic training data for the model

        :param cfg: DictConfig with training configuration parameters
        :param rng: numpy random number generator
        :return: numpy array with the rank-local training data 
        """
        if (cfg.num_samples_per_rank==111):
            samples = 20 * cfg.mini_batch
        else:
            samples = cfg.num_samples_per_rank
        data = np.float32(rng.normal(size=(samples,self.ndIn+self.ndOut)))
        return data
    
    def load_data(self, cfg: DictConfig, comm) -> np.ndarray:
        """"
        Load training data for the model

        :param cfg: DictConfig with training configuration parameters
        :param comm: MPI communication class
        :return: numpy array with the rank-local training data 
        """
        
        extension = cfg.data_path.split(".")[-1]
        if "npy" in extension:
            data = np.float32(np.load(cfg.data_path))
        elif "vtu" in extension or "vtk" in extension:
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(cfg.data_path)
            reader.Update()
            polydata = reader.GetOutput()
            features = VN.vtk_to_numpy(polydata.GetPointData().GetArray("inputs"))
            targets = VN.vtk_to_numpy(polydata.GetPointData().GetArray("outputs"))
            data = np.hstack((features,targets))

        # Scale the outputs to [0,1] range
        if (np.amin(data[:,self.ndIn:]) < 0 or np.amax(data[:,self.ndIn:]) > 1):
            self.scaler_fit = True
            for i in range(self.ndOut):
                ind = self.ndIn+i
                self.min_val[i] = np.amin(data[:,ind])
                self.max_val[i] = np.amax(data[:,ind])
                data[:,ind] = (data[:,ind] - self.min_val[i]) / \
                              (self.max_val[i] - self.min_val[i])
        return data
    
    def setup_dataloaders(self, data: np.ndarray, cfg, comm, logger) -> dict:
        """
        Prepare the training and validation data loaders 

        :param data: training data
        :param cfg: DictConfig with training configuration parameters
        :param comm: MPI communication class
        :return: tuple of DataLoaders 
        """
        # DataSet
        samples = data.shape[0]
        nVal = m.floor(samples*cfg.validation_split)
        nTrain = samples-nVal
        if (nVal==0 and cfg.validation_split>0):
            if (comm.rank==0): logger.warning("Insufficient number of samples for validation -- skipping it")
        dataset = MiniBatchDataset(data)
        trainDataset, valDataset = random_split(dataset, [nTrain, nVal])

        # DataLoader
        # Try:
        # - pin_memory=True - should be faster for GPU training
        # - num_workers > 1 - enables multi-process data loading 
        # - prefetch_factor >1 - enables pre-fetching of data
        if (cfg.data_path == "synthetic"):
            # Each rank has loaded only their part of training data
            train_sampler = None
            val_sampler = None
            val_dataloader = None
            train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                          shuffle=True, drop_last=True) 
            if (nVal>0):
                val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                            drop_last=True)
        else:
            # Each rank has loaded all the training data, so restrict data loader to a subset of dataset
            val_sampler = None
            val_dataloader = None
            train_sampler = DistributedSampler(trainDataset, num_replicas=comm.size, rank=comm.rank,
                                           shuffle=True, drop_last=True) 
            train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                  sampler=train_sampler)
            if (nVal>0):
                val_sampler = DistributedSampler(valDataset, num_replicas=comm.size, rank=comm.rank,
                                                 drop_last=True) 
                val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                            sampler=val_sampler)
                
        return {
            'train': {
                'loader': train_dataloader,
                'sampler': train_sampler,
                'samples': nTrain
            },
            'validation': {
                'loader': val_dataloader,
                'sampler': val_sampler,
                'samples': nVal
            }
        }
    
    def online_scaler(self, comm, data: torch.Tensor) -> torch.Tensor:
        """
        Perform the min-max scaling on the model outputs when online training
        """
        # Compute scaler and send to DB
        if not self.scaler_fit:
            self.scaler_fit = True
            local_mins = torch.amin(data[:,self.ndIn:], 0)
            local_maxs = torch.amax(data[:,self.ndIn:], 0)
            if (comm.size>1):
                local_maxs = torch.div(1., local_maxs)
                tmp_local = torch.vstack((local_mins, local_maxs))
                tmp_global = torch.zeros_like(tmp_local)
                comm.comm.Allreduce(tmp_local, tmp_global, op=comm.min)
                global_mins = tmp_global[0]
                global_maxs = torch.div(1,tmp_global[1])
                self.min_val = global_mins.numpy()
                self.max_val = global_maxs.numpy()
            else:
                self.min_val = local_mins.numpy()
                self.max_val = local_maxs.numpy()

        # Apply scaler
        for i in range(self.ndOut):
            ind = self.ndIn+i
            data[:,ind] = (data[:,ind] - self.min_val[i]) / \
                          (self.max_val[i] - self.min_val[i])
        return data

    def online_dataloader(self, cfg, client, comm, keys: list, logger, shuffle: Optional[bool] = False) \
                        -> Tuple[torch.utils.data.DataLoader, float]:
        """
        Load data from database and create on-rank data loader
        """
        logger.debug(f'[{comm.rank}]: Grabbing tensors with key {keys}')
        
        if (cfg.precision == "fp32" or cfg.precision == "tf32"):
            dtype = torch.float32
        elif (cfg.precision == "fp64"):
            dtype = torch.float64
        elif (cfg.precision == "fp16"):
            dtype = torch.float16
        elif (cfg.precision == "bf16"):
            dtype = torch.bfloat16

        concat_tensor = torch.cat([torch.from_numpy(client.get_array(key, 'train')).type(dtype) \
                                    for key in keys], dim=0)

        data_loader = DataLoader(MiniBatchDataset(concat_tensor), 
                                 shuffle=shuffle, batch_size=cfg.mini_batch)
        return data_loader

    def script_model(self):
        """
        Return a JIT scripted version of the model
        """
        jit_model = torch.jit.script(self)
        return jit_model

    def save_checkpoint(self, fname: str, data: torch.Tensor):
        """
        Save model checkpoint and min-max scaling
        """
        torch.save(self.state_dict(), f"{fname}.pt", _use_new_zipfile_serialization=False)
        jit_model = self.script_model()
        torch.jit.save(jit_model, f"{fname}_jit.pt")

        if self.scaler_fit:
            with open(f"{fname}_scaling.dat", "w") as fh:
                for i in range(self.ndOut):
                    fh.write(f"{self.min_val[i]:>8e} {self.max_val[i]:>8e}\n")

    
            
