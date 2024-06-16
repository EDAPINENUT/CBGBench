
from collections import Counter
from scipy import spatial as sci_spatial
import numpy as np

RING_TYPE_DISTRIBUTION = {3: 0.012974361055980106,
 4: 0.0019932617923974047,
 5: 0.2855064444525153,
 6: 0.6894442581028335,
 7: 0.009763350901667888,
 8: 0.00031832369460580204}

RING_TYPE_RATIO = {3: 0.032995213439314997,
 4: 0.0050690818605106025,
 5: 0.7260739879493853,
 6: 1.7533318482861286,
 7: 0.02482926484730274,
 8: 0.0008095318298135904}

def ring_type_from_mol(mol):
    ring_info = mol.GetRingInfo()
    ring_type = [len(r) for r in ring_info.AtomRings()]
    return ring_type

def eval_ring_type_distribution(pred_counter: Counter):

    total_num_rings = sum(pred_counter.values())
    pred_ring_distribution = {}
    for k in RING_TYPE_DISTRIBUTION:
        pred_ring_distribution[k] = pred_counter[k] / total_num_rings
    # print('pred ring distribution: ', pred_ring_distribution)
    # print('ref  ring distribution: ', ring_TYPE_DISTRIBUTION)
    js = sci_spatial.distance.jensenshannon(np.array(list(RING_TYPE_DISTRIBUTION.values())),
                                            np.array(list(pred_ring_distribution.values())))
    return js, pred_ring_distribution

def eval_ring_type_ratio(ring_type, num_mols):

    pred_ring_ratio = {}
    for k in RING_TYPE_RATIO:
        pred_ring_ratio[k] = ring_type[k] / num_mols

    mae = np.abs((np.array(list(RING_TYPE_RATIO.values())) - 
                    np.array(list(pred_ring_ratio.values())))).mean()
    return mae, pred_ring_ratio