import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import os

BOND_TYPES = frozenset(((6, 6, 1), (6, 6, 2), (6, 6, 4), (6, 7, 1), (6, 7, 2), (6, 7, 4), (6, 8, 1), (6, 8, 2),))

DISTANCE_BINS = np.arange(1.1, 1.7, 0.005)[:-1]

EMPIRICAL_BINS = {
    'CC_2A': np.linspace(0, 2, 100),
    'All_12A': np.linspace(0, 12, 100)
}

PAIR_EMPIRICAL_DISTRIBUTIONS =  np.load(os.path.join(current_dir, '_ref_pairdist_distribution.npy'),
                                  allow_pickle=True).tolist()

EMPIRICAL_DISTRIBUTIONS = np.load(os.path.join(current_dir, '_ref_length_distribution.npy'),
                                  allow_pickle=True).tolist()

assert set(BOND_TYPES) == set(EMPIRICAL_DISTRIBUTIONS.keys())

for v in EMPIRICAL_DISTRIBUTIONS.values():
    assert len(DISTANCE_BINS) + 1 == len(v)
