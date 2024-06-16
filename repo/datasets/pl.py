from .transforms import get_transform
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from tqdm.auto import tqdm
import os
import pickle
import lmdb
from .parsers import *
from easydict import EasyDict   


_DATASET_DICT = {}

def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls
    return decorator


def get_pl_dataset(cfg):
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    dataset = _DATASET_DICT[cfg.name](cfg, transform=transform)
    
    split_by_name = torch.load(cfg.split_path)
    split = {
        k: [dataset.name2id[n] for n in names if n in dataset.name2id]
        for k, names in split_by_name.items()
    }
    subsets = {k: (Subset(dataset, indices=v)) for k, v in split.items()}

    train_set, val_set, test_set = subsets["train"], subsets["test"], subsets["test"]

    return {'train': train_set, 'val': val_set, 'test': test_set}



class LMDataBase(object):
    def __init__(self, processed_path, name2id_path, transform) -> None:
        self.processed_path = processed_path
        self.name2id_path = name2id_path
        self.transform = transform

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.get_pickle_data(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = tuple(data['entry'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def get_pickle_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data
    
    def get_raw(self, idx):
        return self.get_pickle_data(idx)

    


@register_dataset('pl_decomp')
class PocketLigandPairDatasetDecompMol(Dataset, LMDataBase):

    def __init__(self, cfg, transform=None):
        Dataset.__init__(self)

        version = cfg.get('version', 'linker')
        raw_path = cfg.raw_path

        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        
        self.procesed_dir = cfg.get('processed_dir', './data/pl_dcomp/')

        if not os.path.exists(self.procesed_dir):
            os.makedirs(self.procesed_dir)

        self.processed_path = os.path.join(self.procesed_dir,
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.name2id_path = (os.path.join(self.procesed_dir, 'crossdocked_name2id_{}.pt'.format(version)) 
                        if 'crossdocked' in self.raw_path else os.path.join(self.procesed_dir, self.raw_path.split('/')[-1] + '_name2id_{}.pt'.format(version)))

        self.transform = transform

        LMDataBase.__init__(self, 
                            processed_path=self.processed_path, 
                            name2id_path=self.name2id_path,
                            transform=self.transform)

        self.db = None
        
        self.version = version
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        if not os.path.exists(self.name2id_path):
            print(f'{self.name2id_path} does not exist, begin precomputing name2id')
            self._precompute_name2id()
            
        self.name2id = torch.load(self.name2id_path)


        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        data_list = []
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index, dynamic_ncols=True, desc='Parsing protein-ligand pairs')):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProteinFA(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = decomp_parse_sdf_file(os.path.join(data_prefix, ligand_fn), self.version)
                    if ligand_dict is None:
                        raise ValueError('No fragmentated data is available.')
                    data = EasyDict(
                        {'protein': torchify_dict(pocket_dict),
                         'ligand': torchify_dict(ligand_dict)}
                    )
                    data.entry = (pocket_fn, ligand_fn)
                    data_list.append(data)

                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        
        id = 0
        frag_sample_num = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, data in enumerate(tqdm(data_list, dynamic_ncols=True, desc='Writing to LMDB')):
                if data is None:
                    continue
                data['id'] = id
                frag_sample_num += len(data['ligand']['gen_index'])
                txn.put(str(data['id']).encode('utf-8'), pickle.dumps(data))
                id += 1
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_pickle_data(idx)
        # print(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    
@register_dataset('pl_fa')
class PocketLigandPairDatasetFullAtom(Dataset, LMDataBase):

    def __init__(self, cfg, transform=None):
        version = cfg.get('version', 'fullatom')
        raw_path = cfg.raw_path

        Dataset.__init__(self)
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        
        self.procesed_dir = cfg.get('processed_dir', './data/pl/')

        if not os.path.exists(self.procesed_dir):
            os.makedirs(self.procesed_dir)

        self.processed_path = os.path.join(self.procesed_dir,
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.name2id_path = (os.path.join(self.procesed_dir, 'crossdocked_name2id.pt') 
                             if 'crossdocked' in self.raw_path else os.path.join(self.procesed_dir, self.raw_path.split('/')[-1] + '_name2id.pt'))

        self.transform = transform

        LMDataBase.__init__(self, 
                    processed_path=self.processed_path, 
                    name2id_path=self.name2id_path,
                    transform=self.transform)
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        if not os.path.exists(self.name2id_path):
            print(f'{self.name2id_path} does not exist, begin precomputing name2id')
            self._precompute_name2id()
            
        self.name2id = torch.load(self.name2id_path)
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        data_list = []
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index, dynamic_ncols=True, desc='Parsing protein-ligand pairs')):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProteinFA(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = EasyDict(
                        {'protein': torchify_dict(pocket_dict),
                        'ligand':torchify_dict(ligand_dict)}
                    )
                    data.entry = (pocket_fn, ligand_fn)
                    data_list.append(data)

                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        
        id = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, data in enumerate(tqdm(data_list, dynamic_ncols=True, desc='Writing to LMDB')):
                if data is None:
                    continue
                data['id'] = id
                txn.put(str(data['id']).encode('utf-8'), pickle.dumps(data))
                id += 1
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_pickle_data(idx)
        # print(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    

@register_dataset('pl_fg')
class PocketLigandPairDatasetFuncGroup(Dataset, LMDataBase):

    def __init__(self, cfg, transform=None):
        version = cfg.get('version', 'funcgroup')
        raw_path = cfg.raw_path

        Dataset.__init__(self)
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        
        self.procesed_dir = cfg.get('processed_dir', './data/pl_fg/')

        if not os.path.exists(self.procesed_dir):
            os.makedirs(self.procesed_dir)

        self.processed_path = os.path.join(self.procesed_dir,
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.name2id_path = (os.path.join(self.procesed_dir, 'crossdocked_name2id.pt') 
                        if 'crossdocked' in self.raw_path else os.path.join(self.procesed_dir, self.raw_path.split('/')[-1] + '_name2id.pt'))


        self.transform = transform

        LMDataBase.__init__(self, 
            processed_path=self.processed_path, 
            name2id_path=self.name2id_path,
            transform=self.transform)
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        if not os.path.exists(self.name2id_path):
            print(f'{self.name2id_path} does not exist, begin precomputing name2id')
            self._precompute_name2id()
            
        self.name2id = torch.load(self.name2id_path)

        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        data_list = []
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index, dynamic_ncols=True, desc='Parsing protein-ligand pairs')):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProteinFA(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    pocket_dict_frame = parse_biopython_structure_frame(os.path.join(data_prefix, pocket_fn))
                    ligand_dict = parse_sdf_file_to_functional_group_linker(os.path.join(data_prefix, ligand_fn))
                    data = EasyDict(
                        {'protein': {'fg': torchify_dict(pocket_dict_frame),
                                    'linker':torchify_dict(pocket_dict)},
                        'ligand':{'fg':torchify_dict(ligand_dict['fg']),
                                  'linker':torchify_dict(ligand_dict['linker'])}}
                    )
                    data.entry = (pocket_fn, ligand_fn)
                    data_list.append(data)

                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        
        id = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, data in enumerate(tqdm(data_list, dynamic_ncols=True, desc='Writing to LMDB')):
                if data is None:
                    continue
                data['id'] = id
                txn.put(str(data['id']).encode('utf-8'), pickle.dumps(data))
                id += 1
        db.close()
    

    def __getitem__(self, idx):
        data = self.get_pickle_data(idx)
        data['ds_idx'] = idx
        # print(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)





@register_dataset('pl_arfg')
class PocketLigandPairDatasetFuncGroup(Dataset, LMDataBase):

    def __init__(self, cfg, transform=None):
        version = cfg.get('version', 'arfuncgroup')
        raw_path = cfg.raw_path

        Dataset.__init__(self)
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        
        self.procesed_dir = cfg.get('processed_dir', './data/pl_arfg/')

        if not os.path.exists(self.procesed_dir):
            os.makedirs(self.procesed_dir)

        self.processed_path = os.path.join(self.procesed_dir,
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.name2id_path = (os.path.join(self.procesed_dir, 'crossdocked_name2id.pt') 
                             if 'crossdocked' in self.raw_path else os.path.join(self.procesed_dir, self.raw_path.split('/')[-1] + '_name2id.pt'))

        self.transform = transform

        LMDataBase.__init__(self, 
            processed_path=self.processed_path, 
            name2id_path=self.name2id_path,
            transform=self.transform)
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        if not os.path.exists(self.name2id_path):
            print(f'{self.name2id_path} does not exist, begin precomputing name2id')
            self._precompute_name2id()
            
        self.name2id = torch.load(self.name2id_path)

        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        data_list = []
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index, dynamic_ncols=True, desc='Parsing protein-ligand pairs')):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path

                    pocket_dict = PDBProteinFA(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file_moltree(os.path.join(data_prefix, ligand_fn))

                    ligand_dict['moltree'], pocket_dict['contact'], pocket_dict['contact_idx'] = reset_moltree_root(
                        ligand_dict['moltree'],
                        ligand_dict['pos'],
                        pocket_dict['pos'])
                    data = EasyDict(
                        {'protein': torchify_dict(pocket_dict),
                        'ligand':torchify_dict(ligand_dict)}
                    )
                    data.entry = (pocket_fn, ligand_fn)
                    data_list.append(data)

                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        
        id = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, data in enumerate(tqdm(data_list, dynamic_ncols=True, desc='Writing to LMDB')):
                if data is None:
                    continue
                data['id'] = id
                txn.put(str(data['id']).encode('utf-8'), pickle.dumps(data))
                id += 1
        db.close()
    

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_pickle_data(idx)
        # print(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data


def reset_moltree_root(moltree, ligand_pos, protein_pos):
    ligand2 = np.sum(np.square(ligand_pos), 1, keepdims=True)
    protein2 = np.sum(np.square(protein_pos), 1, keepdims=True)
    dist = np.add(np.add(-2 * np.dot(ligand_pos, protein_pos.T), ligand2), protein2.T)
    min_dist = np.min(dist, 1)
    avg_min_dist = []
    for node in moltree.nodes:
        avg_min_dist.append(np.min(min_dist[node.clique]))
    root = np.argmin(avg_min_dist)
    if root > 0:
        moltree.nodes[0], moltree.nodes[root] = moltree.nodes[root], moltree.nodes[0]
    contact_idx = np.argmin(np.min(dist[moltree.nodes[0].clique], 0))
    contact_protein = torch.tensor(np.min(dist, 0) < 4 ** 2)

    return moltree, contact_protein, torch.tensor([contact_idx])