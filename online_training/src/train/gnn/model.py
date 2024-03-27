################################################
######## GNN ###################################
################################################
### Distributed GNN developed by Shivam Barwey at Argonne National Laboratory

import yaml
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try: 
    import torch_geometric
    from torch_geometric.data import Data
    import torch_geometric.utils as utils
    import torch_geometric.nn as tgnn
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Cartesian, Distance
except:
    pass

try:
    import gnn
    from gnn.gnn import mp_gnn
    import gnn.graph_connectivity as gcon
    import gnn.graph_plotting as gplot
except:
    pass

class GNN(nn.Module):
    def __init__(self, train_cfg):
        """
        Distributed Graph Neural Network
        """
        super().__init__()
        # Build model
        self.cfg = self.load_model_config(train_cfg.gnn.gnn_config)
        self.input_channels = train_cfg.gnn.input_channels
        self.output_channels = train_cfg.gnn.output_channels
        self.model = self.build_model(self.input_channels, self.output_channels)

        # Initialize local graph data structures
        self.graph_reduced = None
        self.graph_full = None
        self.idx_full2reduced = None
        self.idx_reduced2full = None

        # Define the loss and accuracy functions
        self.loss_fn = nn.MSELoss()
        self.acc_fn = nn.MSELoss()

    def load_model_config(self, config_path: str) -> dict: 
        """
        Load the config file for the GNN model
        :param config_path: path to model config file
        :return: model config
        """
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Config {config_path} is invalid.")
        return config

    def build_model(self, input_channels: int, output_channels: int) -> nn.Module:
        """
        Build the GNN model
        :param input_channels: Number of input channels
        :param output_channels: Number of output channels
        :return: GNN model
        """
        activation = getattr(F, self.cfg["activation"])
        model = mp_gnn(input_channels, 
                           self.cfg["hidden_channels"], 
                           output_channels, 
                           self.cfg["n_mlp_layers"], 
                           activation)
        return model
    
    def setup_local_graph(self, train_cfg, comm, client):
        """
        Setup the rank-local graph from the mesh data
        :param train_cfg: training config
        :param client: SmartRedis client class
        :param comm: MPI communicator class
        """
        pos_key = 'pos_node_rank_%d_size_%d' %(comm.rank,comm.size)
        edge_key = 'edge_index_rank_%d_size_%d' %(comm.rank,comm.size)
        gid_key = 'global_ids_rank_%d_size_%d' %(comm.rank,comm.size)
        lmask_key = 'local_unique_mask_rank_%d_size_%d' %(comm.rank,comm.size)
        hmask_key = 'halo_unique_mask_rank_%d_size_%d' %(comm.rank,comm.size)

        if train_cfg.online.db_launch:
            while True:
                if (client.client.poll_tensor(hmask_key,0,1)):
                    break
            pos = client.client.get_tensor(pos_key).astype('float32').T
            gli = client.client.get_tensor(gid_key).astype('int64').reshape((-1,1))
            ei = self.client.get_tensor(edge_key).astype('int64').T
            local_unique_mask = np.squeeze(self.client.get_tensor(lmask_key).astype('int64'))
            halo_unique_mask = np.array([])
            if comm.size > 1:
                halo_unique_mask = np.squeeze(self.client.get_tensor(hmask_key).astype('int64'))

            comm.comm.Barrier()
            if (comm.rank == 0):
                print("Loaded graph data")
            
            if (self.cfg["debug"]):
                print(f'[{comm.rank}]: pos shape {pos.shape}')
                np.savetxt(f"/eagle/datascience/balin/Nek/nekRS-ML/examples/periodicHill_gnn/pos_node_{RANK}",pos)
                print(f'[{comm.rank}]: gli shape {gli.shape}')
                np.savetxt(f"/eagle/datascience/balin/Nek/nekRS-ML/examples/periodicHill_gnn/global_ids_node_{RANK}",gli,fmt='%d')
                print(f'[{comm.rank}]: ei shape {ei.shape}')
                np.savetxt("/eagle/datascience/balin/Nek/nekRS-ML/examples/periodicHill_gnn/edge_index_{RANK}",ei.T,fmt='%d')
                print(f'[{comm.rank}]: local_unique_mask shape {local_unique_mask.shape}')
                np.savetxt(f"/eagle/datascience/balin/Nek/nekRS-ML/examples/periodicHill_gnn/local_unique_mask_{RANK}",local_unique_mask,fmt='%d')
                print(f'[{comm.rank}]: halo_unique_mask shape {halo_unique_mask.shape}')
                np.savetxt(f"/eagle/datascience/balin/Nek/nekRS-ML/examples/periodicHill_gnn/halo_unique_mask_{RANK}",halo_unique_mask,fmt='%d')
        else:
            main_path = train_cfg.data_path+"/"
            path_to_pos_full = main_path + pos_key
            path_to_ei = main_path + edge_key
            path_to_glob_ids = main_path + gid_key
            path_to_unique_local = main_path + lmask_key
            path_to_unique_halo = main_path + hmask_key
        
            pos = np.loadtxt(path_to_pos_full, dtype=np.float32)
            gli = np.loadtxt(path_to_glob_ids, dtype=np.int64).reshape((-1,1))
            ei = np.loadtxt(path_to_ei, dtype=np.int64).T
            local_unique_mask = np.loadtxt(path_to_unique_local, dtype=np.int64)
            halo_unique_mask = np.array([])
            if comm.size > 1:
                halo_unique_mask = np.loadtxt(path_to_unique_halo, dtype=np.int64)

        # Make the full graph
        self.graph_full = Data(x = None, edge_index = torch.tensor(ei), pos = torch.tensor(pos), 
                         global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), 
                         halo_unique_mask = torch.tensor(halo_unique_mask))
        self.graph_full.edge_index = utils.remove_self_loops(self.graph_full.edge_index)[0]
        self.graph_full.edge_index = utils.coalesce(self.graph_full.edge_index)
        self.graph_full.edge_index = utils.to_undirected(self.graph_full.edge_index)
        self.graph_full.local_ids = torch.tensor(range(self.graph_full.pos.shape[0]))

        # Get reduced (non-overlapping) graph and indices to go from full to reduced  
        self.graph_reduced, self.idx_full2reduced = gcon.get_reduced_graph(self.graph_full)

        # Get the indices to go from reduced back to full graph  
        self.idx_reduced2full = gcon.get_upsample_indices(self.graph_full, self.graph_reduced, self.idx_full2reduced)

    def training_step(self, batch) -> torch.Tensor:
        """
        Perform a training step

        :param batch: a torch.Tensor containing the batched inputs
        :return: loss for the batch
        """
        output = self.model(batch.x, batch.edge_index, 
                            batch.edge_attr, 
                            batch.pos, batch.batch)
        loss = self.loss_fn(output, batch.y)
        return loss
    
    def validation_step(self, batch) -> torch.Tensor:
        """
        Perform a validation step

        :param batch: a torch.Tensor containing the batched inputs and outputs
        :return: tuple with the accuracy and loss for the batch
        """
        output = self.model(batch.x, batch.edge_index, 
                            batch.edge_attr, 
                            batch.pos, batch.batch)
        error = self.loss_fn(output, batch.y)
        loss = self.loss_fn(output, batch.y)
        return error, loss
        
    def test_step(self, batch, return_loss: Optional[bool] = False) -> torch.Tensor:
        """
        Perform a test step

        :param batch: a tensor containing the batched inputs and outputs
        :param return_loss: whether to compute the loss on the testing data
        :return: tuple with the accuracy and loss for the batch
        """
        output = self.model(batch.x, batch.edge_index, 
                            batch.edge_attr, 
                            batch.pos, batch.batch)
        error = self.loss_fn(output, batch.y)

        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(output, batch)
            return error, loss
        else:
            return error
        
    def create_data(self, cfg, rng) -> np.ndarray:
        """"
        Create synthetic training data for the model

        :param cfg: DictConfig with training configuration parameters
        :param rng: numpy random number generator
        :return: numpy array with the rank-local training data 
        """
        n_nodes = self.graph_reduced.pos.shape[0]
        data = np.float32(rng.normal(size=(n_nodes,self.input_channels+self.output_channels)))
        return data
    
    def load_data(self, cfg, comm) -> np.ndarray:
        """"
        Load training data for the model

        :param cfg: DictConfig with training configuration parameters
        :return: numpy array with the rank-local training data 
        """
        main_path = cfg.data_path+"/"
        x_key = 'x_rank_%d_size_%d' %(comm.rank,comm.size)
        y_key = 'y_rank_%d_size_%d' %(comm.rank,comm.size)

        path_to_x = main_path + x_key
        path_to_y = main_path + y_key
        data_x = np.loadtxt(path_to_x, ndmin=2, dtype=np.float32)
        data_y = np.loadtxt(path_to_y, ndmin=2, dtype=np.float32)
        assert data_x.shape[1] == self.input_channels, \
            f"Created model with {self.input_channels} input channels, but loaded data has {data_x.shape[1]}"
        assert data_y.shape[1] == self.output_channels, \
            f"Created model with {self.output_channels} output channels, but loaded data has {data_y.shape[1]}"

        # Get data in reduced format (non-overlapping)
        data_x_reduced = data_x[self.idx_full2reduced, :]
        data_y_reduced = data_y[self.idx_full2reduced, :]
        data = np.hstack((data_x_reduced, data_y_reduced))
        return data
    
    def setup_dataloaders(self, data: np.ndarray, cfg, comm) -> dict:
        """
        Prepare the training and validation data loaders 

        :param data: training data
        :param cfg: DictConfig with training configuration parameters
        :param comm: MPI communication class
        :return: tuple of DataLoaders 
        """
        # Populate edge_attrs
        cart = Cartesian(norm=False, max_value = None, cat = False)
        dist = Distance(norm = False, max_value = None, cat = True)
        self.graph_reduced = cart(self.graph_reduced) # adds cartesian/component-wise distance
        self.graph_reduced = dist(self.graph_reduced) # adds euclidean distance

        # Create training dataset
        reduced_graph_dict = self.graph_reduced.to_dict()
        data_train_list = []
        data_temp = Data(   
                            x = data[:,:self.input_channels], 
                            y = data[:,self.input_channels:]
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]
        data_train_list.append(data_temp)
        nTrain = len(data_train_list) # 1 for now
        train_dataset = data_train_list
       
        # Create validation dataset -- same as train data for now
        data_valid_list = []
        data_temp = Data(   
                            x = data[:,:self.input_channels], 
                            y = data[:,self.input_channels:]
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]
        data_valid_list.append(data_temp)
        nVal = len(data_valid_list) # 1 for now
        val_dataset = data_valid_list 

        # No need for distributed samplers, each ranks loads it's own data
        train_sampler = None
        val_sampler = None
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.mini_batch, 
                                      shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.mini_batch, 
                                    shuffle=False)  

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
    
    def save_checkpoint(self, fname: str, data: torch.Tensor):
        torch.save(self.model.state_dict(), f"{fname}.pt", _use_new_zipfile_serialization=False)
