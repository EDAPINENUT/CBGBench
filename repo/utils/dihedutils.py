import torch
import torch_geometric as tg
from torch_geometric.utils import degree
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
angle_mask_ref = torch.LongTensor([[0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1]]).to(device)

angle_combos = torch.LongTensor([[0, 1],
                                 [0, 2],
                                 [1, 2],
                                 [0, 3],
                                 [1, 3],
                                 [2, 3]]).to(device)


def get_neighbor_ids(data):
    """
    Takes the edge indices and returns dictionary mapping atom index to neighbor indices
    Note: this only includes atoms with degree > 1
    """
    # start, end = edge_index
    # idxs, vals = torch.unique(start, return_counts=True)
    # vs = torch.split_with_sizes(end, tuple(vals))
    # return {k.item(): v for k, v in zip(idxs, vs) if len(v) > 1}
    neighbors = data.neighbors.pop(0)
    n_atoms_per_mol = data.batch.bincount()
    n_atoms_prev_mol = 0

    for i, n_dict in enumerate(data.neighbors):
        new_dict = {}
        n_atoms_prev_mol += n_atoms_per_mol[i].item()
        for k, v in n_dict.items():
            new_dict[k + n_atoms_prev_mol] = v + n_atoms_prev_mol
        neighbors.update(new_dict)
    return neighbors


def get_neighbor_bonds(edge_index, bond_type):
    """
    Takes the edge indices and bond type and returns dictionary mapping atom index to neighbor bond types
    Note: this only includes atoms with degree > 1
    """
    start, end = edge_index
    idxs, vals = torch.unique(start, return_counts=True)
    vs = torch.split_with_sizes(bond_type, tuple(vals))
    return {k.item(): v for k, v in zip(idxs, vs) if len(v) > 1}


def get_leaf_hydrogens(neighbors, x):
    """
    Takes the edge indices and atom features and returns dictionary mapping atom index to neighbors, indicating true
    for hydrogens that are leaf nodes
    Note: this only works because degree = 1 and hydrogen atomic number = 1 (checks when 1 == 1)
    Note: we use the 5th feature index bc this corresponds to the atomic number
    """
    # start, end = edge_index
    # degrees = degree(end)
    # idxs, vals = torch.unique(start, return_counts=True)
    # vs = torch.split_with_sizes(end, tuple(vals))
    # return {k.item(): degrees[v] == x[v, 5] for k, v in zip(idxs, vs) if len(v) > 1}
    leaf_hydrogens = {}
    h_mask = x[:, 0] == 1
    for k, v in neighbors.items():
        leaf_hydrogens[k] = h_mask[neighbors[k]]
    return leaf_hydrogens


def get_dihedral_pairs(edge_index, data):
    """
    Given edge indices, return pairs of indices that we must calculate dihedrals for
    """
    start, end = edge_index
    degrees = degree(end)
    dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
    dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)

    # # first method which removes one (pseudo) random edge from a cycle
    dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

    # prioritize rings for assigning dihedrals
    dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]
    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)
    keep, sorted_keep = [], []

    if len(dihedral_pairs.shape) == 1:
        dihedral_pairs = dihedral_pairs.unsqueeze(0)

    for pair in dihedral_pairs:
        x, y = pair

        if sorted(pair) in sorted_keep:
            continue

        y_cycle_check = [y in cycle for cycle in cycles]
        x_cycle_check = [x in cycle for cycle in cycles]

        if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle
            cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x)
            keep.extend(cycle_indices)

            sorted_keep.extend([sorted(c) for c in cycle_indices])
            continue

        if any(y_cycle_check):
            cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y)
            keep.append(pair)
            keep.extend(cycle_indices)

            sorted_keep.append(sorted(pair))
            sorted_keep.extend([sorted(c) for c in cycle_indices])
            continue

        keep.append(pair)

    keep = [t.to(device) for t in keep]
    return torch.stack(keep).t()


def batch_distance_metrics_from_coords(coords, mask):
    """
    Given coordinates of neighboring atoms, compute bond
    distances and 2-hop distances in local neighborhood
    """
    d_mat_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

    if coords.dim() == 4:
        two_dop_d_mat = torch.square(coords.unsqueeze(1) - coords.unsqueeze(2) + 1e-10).sum(dim=-1).sqrt() * d_mat_mask.unsqueeze(-1)
        one_hop_ds = torch.linalg.norm(torch.zeros_like(coords[0]).unsqueeze(0) - coords, dim=-1)
    elif coords.dim() == 5:
        two_dop_d_mat = torch.square(coords.unsqueeze(2) - coords.unsqueeze(3) + 1e-10).sum(dim=-1).sqrt() * d_mat_mask.unsqueeze(-1).unsqueeze(1)
        one_hop_ds = torch.linalg.norm(torch.zeros_like(coords[0]).unsqueeze(0) - coords, dim=-1)

    return one_hop_ds, two_dop_d_mat


def batch_angle_between_vectors(a, b):
    """
    Compute angle between two batches of input vectors
    """
    inner_product = (a * b).sum(dim=-1)

    # norms
    a_norm = torch.linalg.norm(a, dim=-1)
    b_norm = torch.linalg.norm(b, dim=-1)

    # protect denominator during division
    den = a_norm * b_norm + 1e-10
    cos = inner_product / den

    return cos


def batch_angles_from_coords(coords, mask):
    """
    Given coordinates, compute all local neighborhood angles
    """
    if coords.dim() == 4:
        all_possible_combos = coords[:, angle_combos]
        v_a, v_b = all_possible_combos.split(1, dim=2)  # does one of these need to be negative?
        angle_mask = angle_mask_ref[mask.sum(dim=1).long()]
        angles = batch_angle_between_vectors(v_a.squeeze(2), v_b.squeeze(2)) * angle_mask.unsqueeze(-1)
    elif coords.dim() == 5:
        all_possible_combos = coords[:, :, angle_combos]
        v_a, v_b = all_possible_combos.split(1, dim=3)  # does one of these need to be negative?
        angle_mask = angle_mask_ref[mask.sum(dim=1).long()]
        angles = batch_angle_between_vectors(v_a.squeeze(3), v_b.squeeze(3)) * angle_mask.unsqueeze(-1).unsqueeze(-1)

    return angles


def batch_local_stats_from_coords(coords, mask):
    """
    Given neighborhood neighbor coordinates, compute bond distances,
    2-hop distances, and angles in local neighborhood (this assumes
    the central atom has coordinates at the origin)
    """
    one_hop_ds, two_dop_d_mat = batch_distance_metrics_from_coords(coords, mask)
    angles = batch_angles_from_coords(coords, mask)
    return one_hop_ds, two_dop_d_mat, angles


def batch_dihedrals(p0, p1, p2, p3, angle=False):

    s1 = p1 - p0
    s2 = p2 - p1
    s3 = p3 - p2

    sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
    cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)

    if angle:
        return torch.atan2(sin_d_, cos_d_ + 1e-10)

    else:
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        return sin_d_/den, cos_d_/den


def batch_vector_angles(xn, x, y, yn):
    uT = xn.view(-1, 3)
    uX = x.view(-1, 3)
    uY = y.view(-1, 3)
    uZ = yn.view(-1, 3)

    b1 = uT - uX
    b2 = uZ - uY

    num = torch.bmm(b1.view(-1, 1, 3), b2.view(-1, 3, 1)).squeeze(-1).squeeze(-1)
    den = torch.linalg.norm(b1, dim=-1) * torch.linalg.norm(b2, dim=-1) + 1e-10

    return (num / den).view(-1, 9)


def von_Mises_loss(a, b, a_sin=None, b_sin=None):
    """
    :param a: cos of first angle
    :param b: cos of second angle
    :return: difference of cosines
    """
    if torch.is_tensor(a_sin):
        out = a * b + a_sin * b_sin
    else:
        out = a * b + torch.sqrt(1-a**2 + 1e-5) * torch.sqrt(1-b**2 + 1e-5)
    return out


def rotation_matrix(neighbor_coords, neighbor_mask, neighbor_map, mu=None):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs
        (n_dihedral_pairs, 4, n_generated_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (n_dihedral_pairs, 4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom
        (n_dihedral_pairs, 4) each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
    """

    if not torch.is_tensor(mu):
        # mu = neighbor_coords.sum(dim=1, keepdim=True) / (neighbor_mask.sum(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-10)
        mu_num = neighbor_coords[~neighbor_map.bool()].view(neighbor_coords.size(0), 3, neighbor_coords.size(2), -1).sum(dim=1)
        mu_den = (neighbor_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) - 1 + 1e-10)
        mu = mu_num / mu_den  # (n_dihedral_pairs, n_model_confs, 10)
        mu = mu.squeeze(1)  # (n_dihedral_pairs, n_model_confs, 10)

    p_Y = neighbor_coords[neighbor_map.bool(), :]
    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h3_1 = torch.cross(p_Y, mu, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h2 = -torch.cross(h1, h3, dim=-1)  # (n_dihedral_pairs, n_model_confs, 10)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def rotation_matrix_v2(neighbor_coords):
    """
    Given predicted neighbor coordinates from model, return rotation matrix
    :param neighbor_coords: y or x coordinates for the x or y center node
        (n_dihedral_pairs, 3)
    :return: rotation matrix (n_dihedral_pairs, 3, 3)
    """

    p_Y = neighbor_coords

    eta_1 = torch.rand_like(p_Y)
    eta_2 = eta_1 - torch.sum(eta_1 * p_Y, dim=-1, keepdim=True) / (torch.linalg.norm(p_Y, dim=-1, keepdim=True)**2 + 1e-10) * p_Y
    eta = eta_2 / torch.linalg.norm(eta_2, dim=-1, keepdim=True)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h3_1 = torch.cross(p_Y, eta, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h2 = -torch.cross(h1, h3, dim=-1)  # (n_dihedral_pairs, n_model_confs, 10)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def rotation_matrix_inf(neighbor_coords, neighbor_mask, neighbor_map):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs (4, n_model_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom (4)
        each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (3, 3)
    """

    mu = neighbor_coords.sum(dim=0, keepdim=True) / (neighbor_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-10)
    mu = mu.squeeze(0)
    p_Y = neighbor_coords[neighbor_map.bool(), :].squeeze(0)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)

    h3_1 = torch.cross(p_Y, mu, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)

    h2 = -torch.cross(h1, h3, dim=-1)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def build_alpha_rotation_inf(alpha, n_model_confs):

    H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(n_model_confs, 1, 1)
    H_alpha[:, 1, 1] = torch.cos(alpha)
    H_alpha[:, 1, 2] = -torch.sin(alpha)
    H_alpha[:, 2, 1] = torch.sin(alpha)
    H_alpha[:, 2, 2] = torch.cos(alpha)

    return H_alpha


def random_rotation_matrix(dim):
    yaw = torch.rand(dim)
    pitch = torch.rand(dim)
    roll = torch.rand(dim)

    R = torch.stack([torch.stack([torch.cos(yaw) * torch.cos(pitch),
                                  torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.cos(
                                      roll),
                                  torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.sin(
                                      roll)], dim=-1),
                     torch.stack([torch.sin(yaw) * torch.cos(pitch),
                                  torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(
                                      roll),
                                  torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(
                                      roll)], dim=-1),
                     torch.stack([-torch.sin(pitch),
                                  torch.cos(pitch) * torch.sin(roll),
                                  torch.cos(pitch) * torch.cos(roll)], dim=-1)], dim=-2)

    return R


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
