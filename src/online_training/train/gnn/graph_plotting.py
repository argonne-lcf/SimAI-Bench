import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.rcParams.update({'font.size': 20})
import torch_geometric.utils as utils

def plot_graph(data, rank, savedir):
    
    print('Plotting connectivity on rank %d...' %(rank))
    ms = 60
    ms = 50
    lw_edge = 1
    lw_marker = 0.1

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection="3d")

    G = utils.to_networkx(data=data)

    # Extract node and edge positions from the layout
    pos = dict(enumerate(np.array(data.pos)))
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Get full node positions 
    pos = data.pos 
    
    # Get only the unique node positions 
    idx_local_unique = torch.nonzero(data.local_unique_mask).squeeze()
    pos_local_unique = pos[idx_local_unique]

    idx_halo_unique = torch.nonzero(data.halo_unique_mask).squeeze()
    pos_halo_unique = pos[idx_halo_unique]

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.1)

    ax.scatter(*pos_local_unique.T, s=ms, ec='black', lw=lw_marker, c='red', alpha=1)
    ax.scatter(*pos_halo_unique.T, s=ms, ec='black', lw=lw_marker, c='black', alpha=1)


    # Format axis 
    xlim = [pos[:,0].min() - 0.2, pos[:,0].max() + 0.2]
    ylim = [pos[:,1].min() - 0.2, pos[:,1].max() + 0.2]
    zlim = [pos[:,2].min() - 0.2, pos[:,2].max() + 0.2]
    
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    #ax.view_init(elev=50, azim=27, roll=100)
    #ax.view_init(elev=50, azim=-45)
    #ax.view_init(elev=0, azim=-90, roll=0)
    ax.set_aspect('equal')

    fig.tight_layout()
    plt.savefig(savedir + '/graph_rank_%d.png' %(rank))
    plt.close()
    print('Connectivity plot for rank %d saved to %s' %(rank, savedir))
    return
