from torch_geometric.nn import MessagePassing
from torch import nn
import torch
from ..common import ShiftedSoftplus
from math import pi as PI

class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, neural_network, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = neural_network
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W
    
    
class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x
    