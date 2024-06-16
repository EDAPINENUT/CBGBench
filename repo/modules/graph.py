import torch
import torch.nn.functional as F

def merge_multiple_adjacency(adj_list, attr_adj_list):
    """
    Merge multiple adjacency matrices into a single adjacency matrix.
    Args:
        adj_list (list): List of adjacency matrices.
        attr_adj_list (list): List of edge types.
    Returns:
        torch.Tensor: Merged adjacency matrix.
    """

    num_nodes = adj_list[0].size(0)
    if len(attr_adj_list[0].shape)  == len(adj_list[0].shape):
        num_class = torch.stack(attr_adj_list).max()
        attr_adj_list = [F.one_hot(attr_adj, num_class) for attr_adj in attr_adj_list]

    num_edge_types = attr_adj_list.shape[-1]

    adj = torch.zeros(num_nodes, num_nodes, device=adj_list[0].device)
    attr_adj = torch.zeros(num_nodes, num_nodes, num_edge_types, device=adj_list[0].device)

    for i, (adj_i, attr_i) in enumerate(adj_list, attr_adj_list):
        adj = torch.logical_or(adj, adj_i)
        attr_adj = attr_adj + attr_i * adj_i.unsqueeze(-1)
        
    return adj, attr_adj

def connect_intra_edge(x, mask, return_type=None, mode='knn', cutoff=10.0):
    x[~mask] = 1e6 # make the dummy nodes far away, so that they won't be selected as neighbors
    dist = torch.cdist(x, x)

    if mode == 'knn':
        cutoff = int(cutoff)
        dist_cut_off, knn = torch.topk(dist, k=cutoff, dim=-1, largest=False)
        adj = torch.zeros_like(dist, dtype=torch.bool)
        adj = adj.scatter_(-1, knn, torch.ones_like(adj, dtype=torch.bool))
        adj_dist = torch.zeros_like(dist, dtype=torch.float)
        adj_dist = adj_dist.scatter_(-1, knn, dist_cut_off)

    elif mode == 'radius':
        cutoff = float(cutoff)
        raise NotImplementedError('TODO: implement radius cutoff')
    else:
        raise ValueError(f'Not supported cutoff mode: {mode}')
    

    x[~mask] = 0.
    mask_2D = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    adj = adj * mask_2D
    adj_dist = adj_dist * mask_2D

    if return_type is not None:
        return_type = int(return_type)
        adj_attr = adj.unsqueeze(-1) * return_type
        return adj, adj_attr, adj_dist
    else:
        return adj, adj_dist
    
def connect_inter_edge(x1, x2, mask1, mask2, return_type=None, mode='knn', cutoff=10.0):
    '''
    x1 is the target node, x2 is the source node
    '''
    # make the dummy nodes far away, so that they won't be selected as neighbors
    x1[~mask1] = 1e6
    x2[~mask2] = 1e6

    dist = torch.cdist(x1, x2)
    if mode == 'knn':
        cutoff = int(cutoff)
        dist_cut_off, knn = torch.topk(dist, k=cutoff, dim=-1, largest=False)
        adj = torch.zeros_like(dist, dtype=torch.bool)
        adj = adj.scatter_(-1, knn, torch.ones_like(adj, dtype=torch.bool))
        adj_dist = torch.zeros_like(dist, dtype=torch.float)
        adj_dist = adj_dist.scatter_(-1, knn, dist_cut_off)

    elif mode == 'radius':
        cutoff = float(cutoff)
        raise NotImplementedError('TODO: implement radius cutoff')
    else:
        raise ValueError(f'Not supported cutoff mode: {mode}')
    mask_2D = mask1.unsqueeze(-1) * mask2.unsqueeze(-2)
    adj = adj * mask_2D
    adj_dist = adj_dist * mask_2D

    x1[~mask1] = 0.
    x2[~mask2] = 0.
    if return_type is not None:
        return_type = int(return_type)
        adj_attr = adj.unsqueeze(-1) * return_type
        return adj, adj_attr, adj_dist
    else:
        return adj, adj_dist


def construct_pairwise_feature(h_target, h_source, adj_matrix):
    """
    Construct pairwise feature between target and source nodes.
    Args:
        h_target (torch.Tensor): Target node features.
        h_source (torch.Tensor): Source node features.
        adj_matrix (torch.Tensor): Adjacency matrix.
    Returns:
        torch.Tensor: Pairwise feature.
    """

    num_nodes_tgt = h_target.size(1)
    num_nodes_src = h_source.size(1)

    h_source = h_source.unsqueeze(2).repeat(1, 1, num_nodes_src, 1)
    h_target = h_target.unsqueeze(1).repeat(1, num_nodes_tgt, 1 , 1)
    h = torch.cat([h_source, h_target], dim=-1)

    adj_matrix = adj_matrix.view(-1, num_nodes_src, num_nodes_tgt)
    h = h * adj_matrix.unsqueeze(-1)

    return h

def transform_pairwise_feature_to_mespas(h_pair, adj_matrix):
    """
    Transform pairwise feature to message passing index
    """
    edge_idx = adj_matrix.to_sparse().indices()
    h_pair = h_pair[edge_idx[0], edge_idx[1], edge_idx[2]]
    return h_pair, edge_idx

def transform_node_feature_to_mespas(h_tgt, h_src, adj_matrix):
    """
    Transform node feature to message passing index
    """
    pass

def construct_pairwise_feature(h_target, h_source, adj_matrix, merge='cat'):
    """
    Construct pairwise feature between target and source nodes.
    Args:
        h_target (torch.Tensor): Target node features.
        h_source (torch.Tensor): Source node features.
        adj_matrix (torch.Tensor): Adjacency matrix.
        merge (str): Merge type. Default: 'cat'.
    Returns:
        torch.Tensor: Pairwise feature.
    """

    num_nodes_tgt = h_target.size(1)
    num_nodes_src = h_source.size(1)
    edge_idx = adj_matrix.to_sparse().indices()

    h_source = h_source[edge_idx[0], edge_idx[2]]
    h_target = h_target[edge_idx[0], edge_idx[1]]

    if merge == 'cat':
        h_source = torch.cat([h_source, h_target], dim=-1)
    elif merge == 'add':
        h_source = h_source + h_target
    elif merge == 'sub':
        h_source = h_source - h_target
    else:
        raise ValueError(f'Unknown merge type: {merge}')    

    return h_source, h_target


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index


# def transform_heteo_to_graph(heteo_graph):
