from torch_geometric.nn import radius_graph, MessagePassing
import torch
from torch import nn
from ..common import GaussianSmearing, MLP, outer_product
from .interaction import *

class SchNet(nn.Module):
    def __init__(self, num_node_types, hidden_channels=128, num_filters=32, num_interactions=6, num_gaussians=50, cutoff=10.0):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        self.embedding = nn.Embedding(num_node_types, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians, fixed_offset=False)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
            
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()


    
    def forward(self, z, pos, batch):
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
        edge_attr = self.distance_expansion(edge_weight)
        
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

#         h = self.lin1(h)
#         h = self.act(h)
#         h = self.lin2(h)
        
        return h
        
        
        
        