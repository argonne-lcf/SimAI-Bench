import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
from typing import List, Tuple
Tensor = torch.Tensor


def get_reduced_graph(data_full: Data) -> Tuple[Data, Tensor]:
    """
    This function takes as input the GLL-based overlapping graph, and eliminates 
    overlapping nodes and edges to produce a REDUCED (non-overlapping) GLL-based graph. 
    This is equivalent to a graph pooling operation -- we are reducing the number of nodes 
    and edges in the input graph such that the output graph has no coincident local nodes. 
    """
    # X: [First isolate local nodes]
    idx_local_unique = torch.nonzero(data_full.local_unique_mask).squeeze()
    idx_halo_unique = torch.nonzero(data_full.halo_unique_mask).squeeze()
    idx_keep = torch.cat((idx_local_unique, idx_halo_unique))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PYGEOM FUNCTION -- this gets the reduced edge_index
    num_nodes = data_full.pos.shape[0]
    perm = idx_keep
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = data_full.edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    edge_index_reduced = torch.stack([row, col], dim=0)
    edge_index_reduced = utils.coalesce(edge_index_reduced)
    edge_index_reduced = utils.to_undirected(edge_index_reduced)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pos_reduced = data_full.pos[idx_keep]
    gid_reduced = data_full.global_ids[idx_keep]
    data_reduced = Data(x = None, pos = pos_reduced, edge_index = edge_index_reduced, global_ids = gid_reduced)
    n_not_halo = len(idx_local_unique)
    n_halo = len(idx_halo_unique)
    data_reduced.local_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.local_unique_mask[:n_not_halo] = 1
    data_reduced.halo_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.halo_unique_mask[n_not_halo:] = 1
    data_reduced.local_ids = torch.tensor(range(data_full.pos.shape[0]))

    return data_reduced, idx_keep


def update_global_ids(data_full: Data, data_reduced: Data, idx_full2reduced: Tensor ) -> Data:
    """
    This function replaces the zero entries in global IDs in the full graph with consecutive negative 
    numbers. This is used to produce the upsampling indices.  
    """
    gid = data_full.global_ids
    zero_indices = torch.where(gid == 0)[0]
    consecutive_negatives = -1 * torch.arange(1, len(zero_indices) + 1)
    gid[zero_indices] = consecutive_negatives
    data_full.global_ids = gid 
    data_reduced.global_ids = gid[idx_full2reduced]
    
    return data_full, data_reduced

def get_upsample_indices(data_full: Data, data_reduced: Data, idx_full2reduced: Tensor) -> Tensor:
    """
    This function produces the indices that allow to go from the reduced (non-overlapping) node 
    representation, to the full (overlapping) node representation. 
    """

    # Update global ids 
    data_full, data_reduced = update_global_ids(data_full, data_reduced, idx_full2reduced)

    # Get global ids 
    gid = data_full.global_ids

    # Get quantities from reduced graph (no coincident nodes)  
    gid_reduced = data_reduced.global_ids

    # Step 3: Sorting 
    # Sort full graph based on global id
    _, idx_sort = torch.sort(gid)
    gid = gid[idx_sort]

    # Sort reduced graph based on global id 
    _, idx_sort_reduced = torch.sort(gid_reduced)
    gid_reduced = gid_reduced[idx_sort_reduced]

    # Step 4: Get the scatter assignments 
    count = 0
    scatter_ids = torch.zeros(data_full.pos.shape[0], dtype=torch.int64)
    scatter_ids[0] = count
    for i in range(1,len(gid)):

        gid_prev = gid[i-1]
        gid_curr = gid[i]

        if (gid_curr > gid_prev):
            count += 1

        scatter_ids[i] = count

    return scatter_ids 

def upsample_node_attributes(x_reduced: Tensor, 
                             gid_reduced: Tensor, lid_reduced: Tensor,
                             gid_full: Tensor, lid_full: Tensor, 
                             idx_reduced2full: Tensor) -> Tensor:
    """ 
    Performs the node upsampling to go back to the coincident representation.
    """

    # Sort full graph based on global id
    _, idx_sort = torch.sort(gid_full)
    gid_full = gid_full[idx_sort]
    lid_full = lid_full[idx_sort]

    # Sort reduced graph based on global id 
    _, idx_sort_reduced = torch.sort(gid_reduced)
    gid_reduced = gid_reduced[idx_sort_reduced]
    lid_reduced = lid_reduced[idx_sort_reduced]
    x_reduced = x_reduced[idx_sort_reduced]

    # Scatter back 
    # -- this is ordered by ascending global ID 
    x_full = x_reduced[idx_reduced2full]

    # Un-sort 
    _, idx_sort = torch.sort(lid_full)
    x_full = x_full[idx_sort]

    return x_full



