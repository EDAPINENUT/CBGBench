from rdkit import Chem
import numpy as np
from repo.utils.protein.constants import aa_name_number, backbone_names
from Bio.PDB import Selection, PDBParser
from Bio.PDB.Residue import Residue
from easydict import EasyDict
from repo.utils.protein.constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names, 
    BBHeavyAtom
)
from .icoord import get_chi_angles, get_backbone_torsions

pdb_parser = PDBParser(QUIET = True)
ptable = Chem.GetPeriodicTable()

class ParsingException(Exception):
    pass

class PDBProteinFA(object):

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in backbone_names)
            self.atom_to_aa_type.append(aa_name_number[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in backbone_names:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(aa_name_number[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in backbone_names:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.long),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = np.zeros([max_num_heavyatoms, 3], dtype=np.float32)
    mask_heavyatom = np.zeros([max_num_heavyatoms, ], dtype=np.bool_)
    element_heavyatom = np.zeros([max_num_heavyatoms, ], dtype=np.int_)

    restype = AA(res.get_resname())
    for idx, (atom_name, element_name) in enumerate(zip(restype_to_heavyatom_names[restype], res.get_atoms())):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = np.array(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            element_heavyatom[idx] = ptable.GetAtomicNumber(element_name.element)
    return pos_heavyatom, mask_heavyatom, element_heavyatom


def parse_biopython_structure_frame_sidechain(pdb_path, unknown_threshold=1.0, max_resseq=None):

    entity = pdb_parser.get_structure(id, pdb_path)[0]
    chains = Selection.unfold_entities(entity, 'C')
    # chains.sort(key=lambda c: c.get_id())
    # chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({
        'chain_id': [], 'chain_nb': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [], 'element_heavyatom': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        'phi': [], 'phi_mask': [],
        'psi': [], 'psi_mask': [],
        'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
    })
    tensor_types = {
        'chain_nb': np.int_,
        'resseq': np.int_,
        'res_nb': np.int_,
        'aa': np.int_,
        'pos_heavyatom': np.stack,
        'mask_heavyatom': np.stack,
        'element_heavyatom': np.stack,

        'phi': np.float32,
        'phi_mask': np.bool_,
        'psi': np.float32,
        'psi_mask': np.bool_,

        'chi': np.stack,
        'chi_alt': np.stack,
        'chi_mask': np.stack,
        'chi_complete': np.bool_,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        chain.atom_to_internal_coordinates()
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, element_heavytaom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.element_heavyatom.append(element_heavytaom)

            # Backbone torsions
            phi, psi, _ = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = np.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None

    if (count_unk / count_aa) >= unknown_threshold:
        return None

    # seq_map = {}
    # for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
    #     seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])

    return data



def parse_biopython_structure_frame(pdb_path, unknown_threshold=1.0, max_resseq=None):
    try:
        parsed = {
        }
        entity = pdb_parser.get_structure(id, pdb_path)[0]
        chains = Selection.unfold_entities(entity, 'C')
        chains.sort(key=lambda c: c.get_id())
        data = EasyDict({
            'chain_id': [],
            'resseq': [], 'icode': [], 'res_nb': [],
            'aa': [],
            'pos_heavyatom': [], 'mask_heavyatom': [],
        })
        array_types = {
            'resseq': np.int_,
            'res_nb': np.int_,
            'aa': np.int_,
            'pos_heavyatom': np.stack,
            'mask_heavyatom': np.stack,
        }

        count_aa, count_unk = 0, 0

        for i, chain in enumerate(chains):
            seq_this = 0   # Renumbering residues
            residues = Selection.unfold_entities(chain, 'R')
            residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
            for _, res in enumerate(residues):
                resseq_this = int(res.get_id()[1])
                if max_resseq is not None and resseq_this > max_resseq:
                    continue

                resname = res.get_resname()
                if not AA.is_aa(resname): continue
                if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
                restype = AA(resname)
                count_aa += 1
                if restype == AA.UNK: 
                    count_unk += 1
                    continue

                # Chain info
                data.chain_id.append(chain.get_id())

                # Residue types
                data.aa.append(restype) # Will be automatically cast to torch.long

                # Heavy atoms
                pos_heavyatom, mask_heavyatom, _ = _get_residue_heavyatom_info(res)
                data.pos_heavyatom.append(pos_heavyatom)
                data.mask_heavyatom.append(mask_heavyatom)

                # Sequential number
                resseq_this = int(res.get_id()[1])
                icode_this = res.get_id()[2]
                if seq_this == 0:
                    seq_this = 1
                else:
                    d_CA_CA = np.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                    if d_CA_CA <= 4.0:
                        seq_this += 1
                    else:
                        d_resseq = resseq_this - data.resseq[-1]
                        seq_this += max(2, d_resseq)

                data.resseq.append(resseq_this)
                data.icode.append(icode_this)
                data.res_nb.append(seq_this)

        if len(data.aa) == 0:
            raise ParsingException('No parsed residues.')

        if (count_unk / count_aa) >= unknown_threshold:
            raise ParsingException(
                f'Too many unknown residues, threshold {unknown_threshold:.2f}.'
            )

        seq_map = {}
        for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
            seq_map[(chain_id, resseq, icode)] = i

        for key, convert_fn in array_types.items():
            data[key] = convert_fn(data[key])

        parsed['pocket_seqmap'] = seq_map
        parsed['pocket'] = data

    except:
        data = None

    return data
