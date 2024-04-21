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

class MP_GNN(torch.nn.Module):
    def __init__(self,
                 input_node_channels: int,
                 input_edge_channels: int,
                 hidden_channels: int,
                 output_node_channels: int,
                 n_mlp_hidden_layers: int,
                 n_messagePassing_layers: int,
                 name: Optional[str] = 'gnn'):
        super().__init__()

        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.name = name

        # ~~~~ node encoder MLP
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP
        self.edge_encoder = MLP(
                input_channels = self.input_edge_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ReLU(),
                )

        # ~~~~ Processor
        self.processor = torch.nn.ModuleList()
        for i in range(self.n_messagePassing_layers):
            self.processor.append(
                          MessagePassingLayer(
                                     channels = self.hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers
                                     )
                                  )

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: torch.LongTensor,
            pos: Tensor) -> Tensor:

        # ~~~~ Compute edge features
        x_send = x[edge_index[0,:].to(torch.long),:]
        x_recv = x[edge_index[1,:].to(torch.long),:]
        pos_send = pos[edge_index[0,:].to(torch.long),:]
        pos_recv = pos[edge_index[1,:].to(torch.long),:]
        e_1 = pos_send - pos_recv
        e_2 = torch.norm(e_1, dim=1, p=2, keepdim=True)
        e_3 = x_send - x_recv
        e = torch.cat((e_1, e_2, e_3), dim=1)

        # ~~~~ Node encoder
        x = self.node_encoder(x)

        # ~~~~ Edge encoder
        e = self.edge_encoder(e)
        
        # ~~~~ Processor
        #for i in range(self.n_messagePassing_layers):
        for _, layer in enumerate(self.processor):
            x = layer(x, e, edge_index)

        # ~~~~ Node decoder
        x = self.node_decoder(x)

        return x

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels': self.input_edge_channels,
             'hidden_channels': self.hidden_channels,
             'output_node_channels': self.output_node_channels,
             'n_mlp_hidden_layers': self.n_mlp_hidden_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'name': self.name}
        return a

    def get_save_header(self) -> str:
        a = self.input_dict()
        header = a['name']

        for key in a.keys():
            if key != 'name':
                header += '_' + str(a[key])

        #for item in self.input_dict():
        return header

class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MLP(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 output_channels: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
                 bias: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        if norm_layer is None:
            self.apply_layer = False
            self.norm_layer = DoNothing()
        else:
            self.apply_layer = True
            self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.ic = [input_channels] + hidden_channels # input channel dimensions for each layer
        self.oc = hidden_channels + [output_channels] # output channel dimensions for each layer

        self.mlp = torch.nn.ModuleList()
        for i in range(len(self.ic)):
            self.mlp.append( torch.nn.Linear(self.ic[i], self.oc[i], bias=bias) )

        self.reset_parameters()
        return

    def forward(self, x: Tensor) -> Tensor:
        #for i in range(len(self.ic)):
        for index, layer in enumerate(self.mlp):
            x = layer(x)
            if index < (len(self.ic) - 1):
                x = self.activation_layer(x)
        x = self.norm_layer(x)
        return x

    def reset_parameters(self):
        for module in self.mlp:
            module.reset_parameters()
        if self.apply_layer:
            self.norm_layer.reset_parameters()
        return

class MessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_hidden_layers: int):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 

        # Edge update MLP 
        self.edge_updater = MLP(
                input_channels = self.channels*3,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        # Node update MLP
        self.node_updater = MLP(
                input_channels = self.channels*2,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        self.reset_parameters()

        return 

    def forward(self,
            x: Tensor,
            e: Tensor,
            edge_index: torch.LongTensor) -> Tensor:

        # ~~~~ Edge update 
        x_send = x[edge_index[0,:].to(torch.long),:]
        x_recv = x[edge_index[1,:].to(torch.long),:]
        e += self.edge_updater(
                torch.cat((x_send, x_recv, e), dim=1)
                )
        
        # ~~~~ Edge aggregation
        edge_agg = self.edge_aggregator(x, edge_index, e)

        # ~~~~ Node update 
        x += self.node_updater(
                torch.cat((x, edge_agg), dim=1)
                )

        return x

    def reset_parameters(self):
        self.edge_updater.reset_parameters()
        self.node_updater.reset_parameters()
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
