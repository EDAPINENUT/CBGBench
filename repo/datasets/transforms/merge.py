from typing import Any
from ._base import register_transform
from torch_geometric.data import Data, HeteroData
import torch

@register_transform('merge')
class MergeKeys(object):

    def __init__(self, keys, to_graph=True, excluded_subkeys=[]):
        super().__init__()
        self.keys = keys
        self.to_graph = to_graph
        self.excluded_keys = excluded_subkeys
        
    def __call__(self, data):
        data_merge = {}
        for key in self.keys:
            graph = data[key]
            for k, v in graph.items():
                if k not in self.excluded_keys:
                    data_merge[key + '_' + k] = v
        if self.to_graph:
            return Data(**data_merge)
        else:
            return data_merge

@register_transform('merge_ctx_gen')
class MergeCtxGen(object):

    def __init__(self) -> None:
        pass

    def __call__(self, data) -> Any:
        data.ligand.atom_type = torch.cat([data.ligand.atom_type, data.ligand_ctx.atom_type]).long()
        data.ligand.pos = torch.cat([data.ligand.pos, data.ligand_ctx.pos]).float()
        data.ligand.element = torch.cat([data.ligand.element, data.ligand_ctx.element]).long()
        data.ligand.ctx_flag = torch.cat([torch.zeros_like(data.ligand.lig_flag), 
                                          torch.ones_like(data.ligand_ctx.lig_flag)]).bool()
        data.ligand.gen_flag = torch.cat([torch.ones_like(data.ligand.lig_flag), 
                                          torch.zeros_like(data.ligand_ctx.lig_flag)]).bool()
        data.ligand.lig_flag = torch.cat([data.ligand.lig_flag, data.ligand_ctx.lig_flag]).bool()
        return data


@register_transform('hetero_merge')
class HeteroMergeKeys(object):
    def __init__(self, keys, ignore_attrs) -> None:
        self.keys = keys
        self.ignore_attrs = ignore_attrs
    
    def __call__(self, data):
        data_merge = {}
        for key in self.keys:
            if 'cross' not in key:
                idx_key_names = []
                graph = data[key]
                graph_in = {}
                for k, v in graph.items():
                    # the edge index is dst -> src
                    if 'index' in k and k not in self.ignore_attrs:
                        if (key, 'to', key) not in data_merge:
                            data_merge[(key, 'to', key)] = {}
                            
                        data_merge[(key, 'to', key)][k] = v[[1,0]]
                        prefix = '_'.join(k.split('_')[:-1])
                        idx_key_names.append(k)
                        for k_edge in graph.keys():
                            if prefix in k_edge and k_edge not in self.ignore_attrs:
                                data_merge[(key, 'to', key)][k_edge] = graph[k_edge]
                                idx_key_names.append(k_edge)

                for k, v in graph.items():
                    if k not in self.ignore_attrs and k not in idx_key_names:
                        graph_in[k] = v

                data_merge[key] = graph_in

            elif 'cross' in key:
                graph = data[key]
                src, dst = key.split('_cross_')
                data_merge[(src, 'to', dst)] = {}
                for k, v in graph.items():
                    # the edge index is dst -> src
                    if 'index' in k:
                        v = v[[1,0]]
                    data_merge[(src, 'to', dst)][k] = v

        data_merge = HeteroData(data_merge)

        return data_merge
