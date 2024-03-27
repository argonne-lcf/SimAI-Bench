################################################
######## AnisoSGS ##############################
################################################
# Anisotropic SGS model for LES developed by Aviral Prakash and John A. Evans at Univ. Colorado Boulder
# A. Prakash, et al. 2023. Invariant Data-Driven Subgrid Stress Modeling on Anisotropic Grids for Large Eddy Simulation. arXiv:2212.00332 [physics.flu-dyn]

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

from datasets import OfflineDataset, MiniBatchDataset


class anisoSGS(nn.Module): 
    def __init__(self, inputDim: Optional[int] = 6, outputDim: Optional[int] = 6, 
                 numNeurons: Optional[int] = 20, numLayers: Optional[int] = 1):
        """
        Initialize the anisoSGS model
    
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
            if not cfg.sgs.comp_model_ins_outs:
                features = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("input123_py")),
                                VN.vtk_to_numpy(polydata.GetPointData().GetArray("input456_py"))))
                targets = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("output123_py")),
                                VN.vtk_to_numpy(polydata.GetPointData().GetArray("output456_py"))))
            else:
                features, targets = self.comp_ins_outs_SGS(polydata)
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
    
    def setup_dataloaders(self, data: np.ndarray, cfg, comm) -> dict:
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
            if (comm.rank==0): print("Insufficient number of samples for validation -- skipping it")
        dataset = OfflineDataset(data)
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
    
    def online_scaler(self, comm, client, data: torch.Tensor) -> torch.Tensor:
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

    def online_dataloader(self, cfg, client, comm, keys: list, shuffle: Optional[bool] = False) \
                        -> Tuple[torch.utils.data.DataLoader, float]:
        """
        Load data from database and create on-rank data loader
        """
        if (cfg.logging=='debug'):
            print(f'[{comm.rank}]: Grabbing tensors with key {keys}', flush=True)
        rtime = perf_counter()
        concat_tensor = torch.cat([torch.from_numpy(client.client.get_tensor(key).astype('float32')) \
                                    for key in keys], dim=0)
        rtime = perf_counter() - rtime

        concat_tensor = self.online_scaler(comm, client, concat_tensor)

        if (cfg.precision == "fp32" or cfg.precision == "tf32"):
            concat_data = torch.tensor(concat_data, dtype=torch.float32)
        elif (cfg.precision == "fp64"):
            concat_data = torch.tensor(concat_data, dtype=torch.float64)
        elif (cfg.precision == "fp16"):
            concat_data = torch.tensor(concat_data, dtype=torch.float16)
        elif (cfg.precision == "bf16"):
            concat_data = torch.tensor(concat_data, dtype=torch.bfloat16)

        data_loader = DataLoader(MiniBatchDataset(concat_tensor), 
                                 shuffle=shuffle, batch_size=cfg.mini_batch)
        return data_loader, rtime

    def save_checkpoint(self, fname: str, data: torch.Tensor):
        """
        Save model checkpoint and min-max scaling
        """
        torch.save(self.state_dict(), f"{fname}.pt", _use_new_zipfile_serialization=False)
        module = torch.jit.trace(self, data)
        torch.jit.save(module, f"{fname}_jit.pt")

        if self.scaler_fit:
            with open(f"{fname}_scaling.dat", "w") as fh:
                for i in range(self.ndOut):
                    fh.write(f"{self.min_val[i]:>8e} {self.max_val[i]:>8e}\n")

    def comp_ins_outs_SGS(self, polydata, alignment="vorticity"):
        """
        Compute the inputs and outputs for the anisotropic SGS model from raw data
        """

        # Read raw data from file
        GradU = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradUFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradVFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradZFilt"))))
        GradU = np.reshape(GradU, (-1,3,3))
        Delta = VN.vtk_to_numpy(polydata.GetPointData().GetArray("gij"))
        SGS = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("SGS_diag")),
                        VN.vtk_to_numpy(polydata.GetPointData().GetArray("SGS_offdiag"))))
        
        # Initialize new arrays
        nsamples = GradU.shape[0]
        Deltaij = np.zeros((3,3))
        Gij = np.zeros((3,3))
        Sij = np.zeros((3,3))
        Oij = np.zeros((3,3))
        vort = np.zeros((3))
        lda = np.zeros((3))
        eigvecs = np.zeros((3,3))
        eigvecs_aligned = np.zeros((3,3))
        vort_Sframe = np.zeros((3))
        inputs = np.zeros((nsamples,6))
        tmp = np.zeros((3,3))
        outputs = np.zeros((nsamples,6))

        # Loop over number of grid points and compute model inputs and outputs
        scaling = [3, 3, 3]
        eps = 1.0e-14
        nu = 1.25e-5
        for i in range(nsamples):
            Deltaij[0,0] = Delta[i,0]*scaling[0]
            Deltaij[1,1] = Delta[i,1]*scaling[1]
            Deltaij[2,2] = Delta[i,2]*scaling[2]
            Deltaij_norm = m.sqrt(Deltaij[0,0]**2 + Deltaij[1,1]**2 + Deltaij[2,2]**2)
            Deltaij = Deltaij / (Deltaij_norm+eps)

            Gij = np.matmul(GradU[i],Deltaij)
            Sij[0,0] = Gij[0,0]
            Sij[1,1] = Gij[1,1]
            Sij[2,2] = Gij[2,2]
            Sij[0,1] = 0.5*(Gij[0,1]+Gij[1,0])
            Sij[0,2] = 0.5*(Gij[0,2]+Gij[2,0])
            Sij[1,2] = 0.5*(Gij[1,2]+Gij[2,1])
            Sij[1,0] = Sij[0,1]
            Sij[2,0] = Sij[0,2]
            Sij[2,1] = Sij[1,2]
            Oij[0,1] = 0.5*(Gij[0,1]-Gij[1,0])
            Oij[0,2] = 0.5*(Gij[0,2]-Gij[2,0])
            Oij[1,2] = 0.5*(Gij[1,2]-Gij[2,1])
            Oij[1,0] = -Oij[0,1]
            Oij[2,0] = -Oij[0,2]
            Oij[2,1] = -Oij[1,2]
            vort[0] = -2*Oij[1,2]
            vort[1] = -2*Oij[0,2]
            vort[2] = -2*Oij[0,1]

            evals, evecs = la.eig(Sij)
            if (alignment=="vorticity"):
                vec = vort.copy()
            elif (alignment=="wall-normal"):
                vec = np.array([0,1,0])
            else:
                print("Alignment option not known, used default vorticity alignment")
                vec = vort.copy()
            lda, eigvecs, eigvecs_aligned = self.align_tensors(evals,evecs,vec)

            Sij_norm = m.sqrt(Sij[0,0]**2+Sij[1,1]**2+Sij[2,2]**2 \
                            + 2*(Sij[0,1]**2+Sij[0,2]**2+Sij[1,2]**2))
            vort_norm = m.sqrt(vort[0]**2 + vort[1]**2 + vort[2]**2)
            SpO = Sij_norm**2 + 0.5*vort_norm**2

            vort_Sframe[0] = np.dot(vort,eigvecs_aligned[:,0])
            vort_Sframe[1] = np.dot(vort,eigvecs_aligned[:,1])
            vort_Sframe[2] = np.dot(vort,eigvecs_aligned[:,2])
            inputs[i,0] = lda[0] / (m.sqrt(SpO)+eps)
            inputs[i,1] = lda[1] / (m.sqrt(SpO)+eps)
            inputs[i,2] = lda[2] / (m.sqrt(SpO)+eps)
            inputs[i,3] = vort_Sframe[0] / (m.sqrt(SpO)+eps)
            inputs[i,4] = vort_Sframe[1] / (m.sqrt(SpO)+eps)
            inputs[i,5] = nu / (Deltaij_norm**2 * m.sqrt(SpO) + eps)

            tmp[0,0] = SGS[i,0] / (Deltaij_norm**2 * SpO + eps)
            tmp[1,1] = SGS[i,1] / (Deltaij_norm**2 * SpO + eps)
            tmp[2,2] = SGS[i,2] / (Deltaij_norm**2 * SpO + eps)
            tmp[0,1] = SGS[i,3] / (Deltaij_norm**2 * SpO + eps)
            tmp[0,2] = SGS[i,4] / (Deltaij_norm**2 * SpO + eps)
            tmp[1,2] = SGS[i,5] / (Deltaij_norm**2 * SpO + eps)
            tmp[1,0] = tmp[0,1]
            tmp[2,0] = tmp[0,2]
            tmp[2,1] = tmp[1,2]
            tmp = np.matmul(np.transpose(eigvecs_aligned),
                            np.matmul(tmp,eigvecs_aligned))
            outputs[i,0] = tmp[0,0]
            outputs[i,1] = tmp[1,1]
            outputs[i,2] = tmp[2,2]
            outputs[i,3] = tmp[0,1]
            outputs[i,4] = tmp[0,2]
            outputs[i,5] = tmp[1,2]
        
        return inputs, outputs

    def align_tensors(evals,evecs,vec):
        """
        Align the eigenvalues and eignevectors according to the local vector (used by comp_ins_outs_SGS)
        """

        if (evals[0]<1.0e-8 and evals[1]<1.0e-8 and evals[2]<1.0e-8):
            index = [0,1,2]
            print("here")
        else:
            index = np.flip(np.argsort(evals))
        lda = evals[index]

        vec_norm = m.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        vec = vec/vec_norm

        eigvec = np.zeros((3,3))
        eigvec[:,0] = evecs[:,index[0]]
        eigvec[:,1] = evecs[:,index[1]]
        eigvec[:,2] = evecs[:,index[2]]

        eigvec_vort_aligned = eigvec.copy()
        if (np.dot(vec,eigvec_vort_aligned[:,0]) < np.dot(vec,-eigvec_vort_aligned[:,0])):
            eigvec_vort_aligned[:,0] = -eigvec_vort_aligned[:,0]
        if (np.dot(vec,eigvec_vort_aligned[:,2]) < np.dot(vec,-eigvec_vort_aligned[:,2])):
            eigvec_vort_aligned[:,2] = -eigvec_vort_aligned[:,2]
        eigvec_vort_aligned[0,1] = (eigvec_vort_aligned[1,2]*eigvec_vort_aligned[2,0]) \
                                - (eigvec_vort_aligned[2,2]*eigvec_vort_aligned[1,0])
        eigvec_vort_aligned[1,1] = (eigvec_vort_aligned[2,2]*eigvec_vort_aligned[0,0]) \
                                - (eigvec_vort_aligned[0,2]*eigvec_vort_aligned[2,0])
        eigvec_vort_aligned[2,1] = (eigvec_vort_aligned[0,2]*eigvec_vort_aligned[1,0]) \
                                - (eigvec_vort_aligned[1,2]*eigvec_vort_aligned[0,0])

        return lda, eigvec, eigvec_vort_aligned
        
            
