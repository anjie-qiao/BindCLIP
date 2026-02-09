# BindCLIP: A Unified Contrastive–Generative Representation Learning Framework for Virtual Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/xxxx/blob/main/LICENSE)

<!-- [[Code](xxxx - Overview)] -->
Official implementation of **BindCLIP**, proposed in the paper  
**“BindCLIP: A Unified Contrastive–Generative Representation Learning Framework for Virtual Screening”**.

# Requirements

We follow the environment setup of [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)

## Data

We follow the DrugCLIP data setup for training and evaluation (LMDB format), including training data (PDBBind with HomoAug augmentation), and evaluation data for virtual screening benchmarks (e.g., DUD-E / LIT-PCBA).

All data are available at:
https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing

## Train

bash bindclip.sh

## Test

bash test.sh


## Retrieval 

bash retrieval.sh

In the google drive folder, you can find example file for pocket.lmdb and mols.lmdb under retrieval dir.
