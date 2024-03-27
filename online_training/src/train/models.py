##### 
##### Wrapper functions for the avaliable models to initialize them
##### and load/create their required data structures 
#####
from typing import Tuple
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from utils import count_weights

from sgs.model import anisoSGS
from quadconv_core.model import QuadConv
from gnn.model import GNN


def load_model(cfg: DictConfig, comm, client, rng, t_data) -> Tuple[nn.Module, np.ndarray]: 
    """ 
    Return the selected model and its training data

    :param cfg: DictConfig with training configuration parameters
    :param comm: Class containing the MPI communicator information
    :param rng: numpy random number generator
    :return: touple with the model and the data
    """

    # Instantiate model
    if (cfg.model=="sgs"):
        model  = anisoSGS(inputDim=cfg.sgs.inputs, outputDim=cfg.sgs.outputs,
                          numNeurons=cfg.sgs.neurons, numLayers=cfg.sgs.layers)
    elif (cfg.model=="quadconv"):
        model = QuadConv(cfg, client, t_data)
        if (comm.rank==0):
            print('Quad-Conv model with configuration:')
            print(model.cfg)
            print("")
    elif (cfg.model=="gnn"):
        model = GNN(cfg)
        model.setup_local_graph(cfg, comm, client)
    
    n_params = count_weights(model)
    if (comm.rank == 0):
        print(f"\nLoaded {cfg.model} model with {n_params} trainable parameters \n")

    # Load/Generate training data
    if not cfg.online.db_launch:
        if (cfg.data_path == "synthetic"):
            data = model.create_data(cfg, rng)
        else:
            data = model.load_data(cfg, comm)
    else:
        data = np.array([0])

    return model, data 



