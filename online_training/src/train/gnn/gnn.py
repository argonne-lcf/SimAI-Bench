from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric.nn as tgnn
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.typing import Adj, OptTensor, PairTensor
    from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x
except:
    pass

class mp_gnn(torch.nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 hidden_channels: int, 
                 output_channels: int, 
                 n_mlp_layers: List[int], 
                 activation: Callable,
                 spatial_dim: int):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels 
        self.n_mlp_layers = list(n_mlp_layers)
        self.act = activation
        self.spatial_dim = spatial_dim

        # ~~~~ node encoder 
        self.node_encoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == 0:
                input_features = self.input_channels
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_encoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ node decoder 
        self.node_decoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == self.n_mlp_layers[0] - 1:
                input_features = self.hidden_channels
                output_features = self.output_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_decoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ message passing layer 
        self.mp_layer = mp_layer(channels = hidden_channels,
                                 n_mlp_layers_edge = self.n_mlp_layers[1], 
                                 n_mlp_layers_node = self.n_mlp_layers[2],
                                 activation = self.act,
                                 spatial_dim = self.spatial_dim)
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: torch.LongTensor,
            edge_attr: Tensor,
            pos: Tensor,
            batch: Optional[torch.LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Node Encoder: 
        n_layers = self.n_mlp_layers[0]
        #for i in range(n_layers):
        for index, encoder in enumerate(self.node_encoder):
            #x = self.node_encoder[i](x)
            x = encoder(x)
            if index < n_layers - 1:
                x = self.act(x)

        # ~~~~ Message passing: 
        x = self.mp_layer(x, edge_index, edge_attr, pos, batch)
        
        # ~~~~ Node decoder:
        n_layers = self.n_mlp_layers[0]
        #for i in range(n_layers):
        for index, decoder in enumerate(self.node_decoder):
            #x = self.node_decoder[i](x)
            x = decoder(x)
            if index < n_layers - 1:
                x = self.act(x)

        return x 

    def reset_parameters(self):
        for module in self.node_encoder:
            module.reset_parameters()

        for module in self.node_decoder:
            module.reset_parameters()

        self.mp_layer.reset_parameters()
            
        return

    def input_dict(self) -> dict:
        a = {'input_channels': self.input_channels,
             'hidden_channels': self.hidden_channels,
             'output_channels': self.output_channels,
             'n_mlp_layers': self.n_mlp_layers,
             'activation': self.act}
        return a

 
class mp_layer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_layers_edge: int, 
                 n_mlp_layers_node: int,
                 activation: Callable,
                 spatial_dim: int):
        super().__init__()

        self.edge_aggregator = EdgeAggregation()
        self.channels = channels
        self.n_mlp_layers_edge = n_mlp_layers_edge
        self.n_mlp_layers_node = n_mlp_layers_node
        self.act = activation
        self.spatial_dim = spatial_dim

        self.edge_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_edge):
            if j == 0:
                input_features = self.channels*3 + (self.spatial_dim+1) # extra 4 dims comes from edge_attr
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.edge_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.node_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_node):
            if j == 0:
                input_features = self.channels*2
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.node_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: torch.LongTensor,
            edge_attr: Tensor,
            pos: Tensor,
            batch: Optional[torch.LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update 
        x_nei = x[edge_index[0,:], :] 
        x_own = x[edge_index[1,:], :] 
        ea = torch.cat((edge_attr, x_nei, x_own, x_nei - x_own), dim=1)
        n_layers = self.n_mlp_layers_edge
        #for j in range(n_layers):
        #    ea = self.edge_updater[j](ea)
        for index, updater in enumerate(self.edge_updater):
            ea = updater(ea) 
            if index < n_layers - 1:
                ea = self.act(ea)

        edge_agg = self.edge_aggregator(x, edge_index, ea)

        x = torch.cat((x, edge_agg), dim=1)
        n_layers = self.n_mlp_layers_node
        #for j in range(n_layers):
        #    x = self.node_updater[j](x) 
        for index, updater in enumerate(self.node_updater):
            x = updater(x) 
            if index < n_layers - 1:
                x = self.act(x)

        return x  

    def reset_parameters(self):
        for module in self.edge_updater:
            module.reset_parameters()

        for module in self.node_updater:
            module.reset_parameters()
        return

class EdgeAggregation(MessagePassing):
    r"""This is a custom class that returns node quantities that represent the neighborhood-averaged edge features.
    Args:
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes: 
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or 
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`, 
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    propagate_type = {'x': Tensor, 'edge_attr': Tensor}

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
