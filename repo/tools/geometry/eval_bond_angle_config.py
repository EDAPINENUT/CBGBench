import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

EMPIRICAL_DISTRIBUTIONS = np.load(os.path.join(current_dir, '_ref_angle_distribution.npy'),
                                  allow_pickle=True).tolist()
