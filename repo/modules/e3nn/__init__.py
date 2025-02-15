from .unitransformer import *
from .gvptransformer import *
from .itatransformer import *

def get_e3_gnn(cfg, num_classes=None, num_edge_classes=None):
    if num_classes is not None:
        cfg.num_classes = num_classes
    if num_edge_classes is not None:
        cfg.num_edge_classes = num_edge_classes

    if cfg.type == 'unitransformer':
        return UniTransformer(cfg)
    elif cfg.type == 'gvptransformer':
        return GVPTransformer(cfg)
    elif cfg.type == 'ipatransformer':
        return IPATransformer(cfg)
    else:
        raise ValueError(f'Unknown model type: {cfg.type}')
