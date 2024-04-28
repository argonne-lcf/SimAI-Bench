import sys
import os, os.path
from typing import Optional
from time import perf_counter
import numpy as np
import torch
try:
    from torch_geometric.nn import knn_graph
except:
    pass

import dragon

# Dragon Client Class for the Simulation (Data Producer)
class Dragon_Sim_Client:
    def __init__(self, args, rank: int, size: int):
        self.client = None
        self.db_launch = args.db_launch
        self.db_nodes = args.db_nodes
        self.rank = rank
        self.size = size
        self.ppn = args.ppn
        self.ow = True if args.problem_size=="debug" else False
        self.max_mem = args.db_max_mem_size*1024*1024*1024
        self.db_address = None
        self.model = args.model

        self.times = {
            "init": 0.,
            "tot_meta": 0.,
            "tot_train": 0.,
            "train": [],
            "tot_infer": 0.,
            "infer": []
        }
        self.time_stats = {}

        if (self.db_launch == "colocated"):
            self.db_nodes = 1
            self.head_rank = self.ppn * self.rank/self.ppn
        elif (self.db_launch == "clustered"):
            self.ppn = size
            self.head_rank = 0

        # For GNN inference
        self.coords = None
        self.edge_index = None

    


# Dragon Client Class for Training
class Dragon_Train_Client:
    def __init__(self, cfg, rank: int, size: int):
        self.rank = rank
        self.size = size
        self.db_launch = cfg.online.smartredis.db_launch
        self.db_nodes = cfg.online.smartredis.db_nodes
        self.ppn = cfg.ppn
        self.ppd = cfg.ppd
        self.global_shuffling = cfg.online.global_shuffling
        self.batch = cfg.online.batch

        self.client = None
        self.npts = None
        self.ndTot = None
        self.ndIn = None
        self.ndOut = None
        self.num_tot_tensors = None
        self.num_db_tensors = None
        self.head_rank = None
        self.tensor_batch = None
        self.dataOverWr = None

        self.times = {
            "init": 0.,
            "tot_meta": 0.,
            "tot_train": 0.,
            "train": [],
        }
        self.time_stats = {}
        self.train_array_sz = 0


