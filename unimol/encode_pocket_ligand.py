#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    root_dir = args.save_dir

    # all_mols = []
    # for dirpath, dirnames, filenames in os.walk(root_dir):
    #     for f in filenames:
    #         if f.endswith('_mols.lmdb'):
    #             all_mols.append(os.path.join(dirpath, f))

    # for mol_path in tqdm(all_mols, desc="Encoding pocket-ligand pairs"):
    #     mol_files = [f for f in filenames if f.endswith('_mols.lmdb')]

    #     dirpath = os.path.dirname(mol_path)
    #     mol_file = os.path.basename(mol_path)
    #     prefix = mol_file.replace('_mols.lmdb', '')
    #     pocket_file = prefix + '_pocket10_pocket.lmdb'
    #     pocket_path = os.path.join(dirpath, pocket_file)

    #     try:
    #         # Encode ligand + pocket
    #         ligand_emb, pocket_emb = task.encode_pocket_and_ligand(
    #             model,
    #             mol_path,
    #             pocket_path,
    #         )
    #     except Exception as e:
    #         logger.error(f"Error encoding {mol_path} / {pocket_path}: {e}")
    #         continue
        
    #     save_dir = dirpath
    #     os.makedirs(save_dir, exist_ok=True)

    #     lig_path = os.path.join(save_dir, mol_file.replace('_mols.lmdb', '') +  "_ligand_emb.pt")
    #     pkt_path = os.path.join(save_dir, pocket_file.replace('_pocket.lmdb', '') + "_pocket_emb.pt")

    #     torch.save(ligand_emb, lig_path)
    #     torch.save(pocket_emb, pkt_path)
    ligand_lmdbs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".lmdb"):
                ligand_lmdbs.append(os.path.join(dirpath, f))
    logger.info(f"Found {len(ligand_lmdbs)} ligand lmdb files under {root_dir}")
    for lig_path in tqdm(ligand_lmdbs, desc="Encoding ligands"):
        try:
            ligand_emb = task.encode_ligand_cls(
                model,
                lig_path,
            )
        except Exception as e:
            logger.error(f"Error encoding ligand {lig_path}: {e}")
            continue

        save_dir = os.path.dirname(lig_path)
        os.makedirs(save_dir, exist_ok=True)

        prefix = os.path.splitext(os.path.basename(lig_path))[0]
        out_path = os.path.join(save_dir, prefix + "_ligand_emb.pt")

        torch.save(ligand_emb, out_path)
        logger.info(f"Saved ligand embedding to {out_path}")



def cli_main():
    # add args
    parser = options.get_validation_parser()
    parser.add_argument("--save-dir", type=str, default="", help="path for saved embedding data")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
