from ._base import *
import copy
from .molecule_featurizer import *
from .protein_featurizer import *
from .translation import *
from .merge import *
from .init_lig import *
from .edge_constructor import *
from .mask import *
from .focal_builder import *
from .contrastive_sampler import *
from .select import *
from .permutate import *
from .sequential_sampler import *

def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
