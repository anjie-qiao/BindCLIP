# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class ComplexCenterDataset(BaseWrapperDataset):
    """     
    compute COM for duffsion
    """

    def __init__(
        self,
        dataset,
        lig_coord_key="coordinates",
        pocket_coord_key="pocket_coordinates",
        out_lig_key="lig_coord_com",
        out_pocket_key="pocket_coord_com",
    ):
        self.dataset = dataset
        self.lig_coord_key = lig_coord_key
        self.pocket_coord_key = pocket_coord_key
        self.out_lig_key = out_lig_key
        self.out_pocket_key = out_pocket_key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):

        dd = self.dataset[index].copy()

        lig = dd[self.lig_coord_key]        # (N_lig, 3)
        poc = dd[self.pocket_coord_key]     # (N_poc, 3)


        # if lig.size == 0 and poc.size == 0:
        #     return dd

        # if lig.size == 0:
        #     all_pos = poc
        # elif poc.size == 0:
        #     all_pos = lig
        # else:
        #     all_pos = np.concatenate([lig, poc], axis=0)   # (N_lig+N_poc, 3)
        #using pocket coordinates to compute COM
        all_pos = poc
        center = all_pos.mean(axis=0, keepdims=True)       # (1, 3)

        lig_centered = lig - center
        poc_centered = poc - center

        dd[self.out_lig_key] = lig_centered.astype(np.float32)
        dd[self.out_pocket_key] = poc_centered.astype(np.float32)

        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
