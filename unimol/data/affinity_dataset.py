# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils


class AffinityDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        decoy_atoms,
        decoy_coordinates,
        docked_coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.decoy_atoms = decoy_atoms
        self.decoy_coordinates = decoy_coordinates
        self.docked_coordinates = docked_coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]

        # docking  
        docked_coordinates = self.dataset[index][self.docked_coordinates]
        # # fingerprints
        scaffold_fp = self.dataset[index]["scaffold_fp"]
        property_vec = self.dataset[index]["scaffold_fp"]

        # # decoy using pool
        # item = self.dataset[index]
        # pool_atoms = item["hard_decoy_atoms"]            # list, len = 10
        # pool_coords = item["hard_decoy_coordinates"]     # list of conf-list
        # pool_fp    = item["hard_decoy_scaffold_fp"]  # optional list
        # pool_size = len(pool_atoms)
        # if self.is_train:
        #     with data_utils.numpy_seed(self.seed, epoch, index):
        #         pool_idx = np.random.randint(pool_size)
        # else:
        #     with data_utils.numpy_seed(self.seed, 1, index):
        #         pool_idx = np.random.randint(pool_size)
        # decoy_atoms = np.array(pool_atoms[pool_idx])
        # decoy_num_conf = len(pool_coords[pool_idx])
        # if self.is_train:
        #     with data_utils.numpy_seed(self.seed, epoch, index, pool_idx):
        #         decoy_conf_idx = np.random.randint(decoy_num_conf)
        # else:
        #     with data_utils.numpy_seed(self.seed, 1, index, pool_idx):
        #         decoy_conf_idx = np.random.randint(decoy_num_conf)

        # decoy_coordinates = pool_coords[pool_idx][decoy_conf_idx]
        # decoy_scaffold_fp = pool_fp[pool_idx]
        # decoy_property_vec = decoy_scaffold_fp
        
        decoy_atoms = np.array(self.dataset[index][self.decoy_atoms])
        decoy_num_conf = len(self.dataset[index][self.decoy_coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                decoy_conf_idx = np.random.randint(decoy_num_conf)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                decoy_conf_idx = np.random.randint(decoy_num_conf)
        decoy_coordinates = self.dataset[index][self.decoy_coordinates][decoy_conf_idx]
        decoy_scaffold_fp = self.dataset[index]["hard_decoy_scaffold_fp"]
        decoy_property_vec = self.dataset[index]["hard_decoy_scaffold_fp"]
        # decoy_scaffold_fp = self.dataset[index]["decoy_scaffold_fp"]
        # decoy_property_vec = self.dataset[index]["decoy_scaffold_fp"]

        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        hard_label = self.dataset[index]["hard_is_random"]
        pocket = self.dataset[index][self.pocket]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),

            "decoy_atoms": decoy_atoms,
            "decoy_coordinates": np.array(decoy_coordinates, dtype=np.float32),
            "docked_coordinates": docked_coordinates.astype(np.float32),
            "scaffold_fp": scaffold_fp,
            "decoy_scaffold_fp": decoy_scaffold_fp,
            "property_vec": property_vec,
            "decoy_property_vec": decoy_property_vec,
            'hard_label':hard_label,

            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityAugDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket_id"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        #mol_atoms_list = self.dataset[index][self.atoms]
        with data_utils.numpy_seed(self.seed, epoch, index):
            mol_idx = np.random.randint(len(self.dataset[index][self.atoms]))
        atoms = np.array(self.dataset[index][self.atoms][mol_idx])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates][mol_idx])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        #print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][mol_idx][sample_idx]


        #pocket_list = self.dataset[index][self.pocket_atoms]
        with data_utils.numpy_seed(self.seed, epoch, index):
            pocket_idx = np.random.randint(len(self.dataset[index][self.pocket_atoms]))
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms][pocket_idx]]
        )

        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates][pocket_idx])

        smi = self.dataset[index]["smiles"][mol_idx]
        pocket = self.dataset[index][self.pocket][0]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityHNSDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        atoms_hns,
        coordinates_hns,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.atoms_hns = atoms_hns
        self.coordinates_hns = coordinates_hns
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        #print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        atoms_hns = np.array(self.dataset[index][self.atoms_hns])
        coordinates_hns = self.dataset[index][self.coordinates_hns][0]



        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "atoms_hns": atoms_hns,
            "coordinates_hns": coordinates_hns.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class AffinityTestDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity=None,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        #print(len(self.dataset[index][self.pocket_coordinates]))
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        affinity = self.dataset[index][self.affinity]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity.astype(np.float32),
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityMolDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        is_train=False,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        #print(self.dataset[index])
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        #print(size)
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        # check coordinates is 2 dimension or not
        if len(self.dataset[index][self.coordinates][sample_idx].shape) == 2:
            coordinates = self.dataset[index][self.coordinates][sample_idx]
        else:
            coordinates = self.dataset[index][self.coordinates]
        #coordinates = self.dataset[index][self.coordinates][sample_idx]
        #coordinates = self.dataset[index][self.coordinates]
        
        smi = self.dataset[index]["smi"]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])
        if self.pocket in self.dataset[index]:
            pocket = self.dataset[index][self.pocket]
        else:
            pocket = ""
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "pocket": pocket,
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class AffinityValidDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]

        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)