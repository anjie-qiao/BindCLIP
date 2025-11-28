# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore

from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture

from .common import compose_context, ShiftedSoftplus
from .diffusion_utils import cosine_beta_schedule,to_torch_const,index_to_log_onehot,q_v_sample,diffusion_loss, get_beta_schedule, log_1_min_a, sample_time
from .uni_transformer import UniTransformerO2TwoUpdateGeneral

logger = logging.getLogger(__name__)



@register_model("drugclip")
class BindingAffinityModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )


    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        drugclip_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        )
        
        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(14))
        

        
        self.pocket_project = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        
        self.fuse_project = NonLinearHead(
            256, 1, "relu"
        )
        self.classification_head = nn.Sequential(
            nn.Linear(args.pocket.encoder_embed_dim + args.pocket.encoder_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        

        ### diffsion config
        self.pocket_num_class = 9
        self.mol_num_class = 30
        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=1,
            num_layers=7,
            hidden_dim=128,
            n_heads=16,
            k=32,
            edge_feat_dim=4,
            num_r_gaussian=20,
            num_node_types=self.mol_num_class,
            act_fn='relu',
            norm=True,
            cutoff_mode='knn',
            ew_net_type='global',
            num_x2h=1,
            num_h2x=1,
            r_max=10.,
            x2h_out_fc=False,
            sync_twoup=False
        )
        self.ligand_atom_emb = nn.Linear(self.mol_num_class + 1, 127)
        self.protein_atom_emb = nn.Linear(self.pocket_num_class, 127)
        self.v_inference = nn.Sequential(
            nn.Linear(128, 128),  #hidden dim = 128
            ShiftedSoftplus(),
            nn.Linear(128, self.mol_num_class),
        )

        betas = get_beta_schedule(
                beta_schedule='sigmoid',
                beta_start=1.e-7,
                beta_end=2.e-3,
                num_diffusion_timesteps=1000,
            )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)

        alphas_v = cosine_beta_schedule(1000, 0.01)
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,

        #prepare for compact graph
        lig_coord_com,
        pocket_coord_com,
        mol_len,
        pocket_len,
        
        smi_list=None,
        pocket_list=None,
        encode=False,
        masked_tokens=None,
        features_only=True,
        is_train=True,
        **kwargs
    ):
        def get_dist_features(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias

        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(
            mol_src_distance, mol_src_edge_type, "mol"
        )
        mol_outputs = self.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias
        )
        mol_encoder_rep = mol_outputs[0]
        encoder_pair_rep = mol_outputs[1]

        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = get_dist_features(
            pocket_src_distance, pocket_src_edge_type, "pocket"
        )
        pocket_outputs = self.pocket_model.encoder(
            pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias
        )
        pocket_encoder_rep = pocket_outputs[0]


        ####################
        # using ligand emb as condition to help doffusion
        # remove BOS and EOS
        device = mol_src_tokens.device
        B, Lm = mol_src_tokens.shape
        _, Lp = pocket_src_tokens.shape

        mol_tokens = mol_src_tokens[:, 1:-1]          # (B, N_lig)
        pocket_tokens = pocket_src_tokens[:, 1:-1]    # (B, N_poc)
        lig_coord = lig_coord_com[:, 1:-1, :].to(dtype=self.ligand_atom_emb.weight.dtype)         # (B, N_lig, 3)
        pocket_coord = pocket_coord_com[:, 1:-1, :].to(dtype=self.protein_atom_emb.weight.dtype)

        idx_lig = torch.arange(Lm-2, device=device).unsqueeze(0)        # (1, Lm-2)
        lig_mask = idx_lig < mol_len.unsqueeze(1)                              # (B, Lm-2)
        idx_poc = torch.arange(Lp-2, device=device).unsqueeze(0)        # (1, Lp-2)
        poc_mask = idx_poc < pocket_len.unsqueeze(1)    

        mol_tokens_flat = mol_tokens[lig_mask]           # (N_lig_total,)
        lig_coord_flat = lig_coord[lig_mask]             # (N_lig_total, 3)
        pocket_tokens_flat = pocket_tokens[poc_mask]     # (N_poc_total,)
        pocket_coord_flat = pocket_coord[poc_mask]       # (N_poc_total, 3) 

        batch_ids = torch.arange(B, device=device)
        batch_ligand = torch.repeat_interleave(batch_ids, mol_len, dim=0)      # (N_lig_total,)
        batch_protein = torch.repeat_interleave(batch_ids, pocket_len, dim=0)  # (N_poc_total,)

        # forward add noise
        time_step, _ = sample_time(B, num_timesteps=1000, device=device)
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(lig_coord_flat)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        lig_coord_perturbed = a_pos.sqrt() * lig_coord_flat + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(mol_tokens_flat, self.mol_num_class)
        ligand_v_perturbed, log_ligand_vt = q_v_sample(log_ligand_v0, self.log_alphas_cumprod_v, self.log_one_minus_alphas_cumprod_v, time_step, batch_ligand)
        ligand_v_perturbed = F.one_hot(ligand_v_perturbed, self.mol_num_class)
        #concat timestep
        ligand_v_perturbed = torch.cat([ligand_v_perturbed,
                    (time_step / 1000)[batch_ligand].unsqueeze(-1)
                ], -1).to(dtype=self.ligand_atom_emb.weight.dtype)
        # predict x0
        h_mol = self.ligand_atom_emb(ligand_v_perturbed)

        pocket_v0 = F.one_hot(pocket_tokens_flat, num_classes=self.pocket_num_class).to(dtype=self.protein_atom_emb.weight.dtype)
        h_pocket = self.protein_atom_emb(pocket_v0)

        h_pocket = torch.cat([h_pocket, torch.zeros(len(h_pocket), 1).to(h_pocket)], -1)
        h_mol = torch.cat([h_mol, torch.ones(len(h_mol), 1).to(h_mol)], -1)
        
        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_pocket,
            h_ligand=h_mol,
            pos_protein=pocket_coord_flat,
            pos_ligand=lig_coord_perturbed,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        ligand_emb =  mol_encoder_rep[:,1:-1,:]
        ligand_emb_flat = ligand_emb[lig_mask] # alige atom 
 
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, ligand_emb_flat, return_all=False, fix_x=False)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)
        loss_diffusion= diffusion_loss(ligand_pos=lig_coord_flat,ligand_pos_perturbed=lig_coord_perturbed,pred_ligand_pos=final_ligand_pos,
        pred_ligand_v=final_ligand_v,log_ligand_v0=log_ligand_v0,log_ligand_vt=log_ligand_vt,
        log_alphas_cumprod_v=self.log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v=self.log_one_minus_alphas_cumprod_v, 
        log_alphas_v=self.log_alphas_v, log_one_minus_alphas_v=self.log_one_minus_alphas_v, time_step=time_step,batch_ligand=batch_ligand)

        ####################


        mol_rep =  mol_encoder_rep[:,0,:]
        pocket_rep = pocket_encoder_rep[:,0,:]

        mol_emb = self.mol_project(mol_rep)
        mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
        pocket_emb = self.pocket_project(pocket_rep)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
                

        ba_predict = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))

        
        # mask duplicate mols and pockets in same batch
        
        bsz = ba_predict.shape[0]
        
        pockets = np.array(pocket_list, dtype=str)
        pockets = np.expand_dims(pockets, 1)
        matrix1 = np.repeat(pockets, len(pockets), 1)
        matrix2 = np.repeat(np.transpose(pockets), len(pockets), 0)
        pocket_duplicate_matrix = matrix1==matrix2
        pocket_duplicate_matrix = 1*pocket_duplicate_matrix
        pocket_duplicate_matrix = torch.tensor(pocket_duplicate_matrix, dtype=ba_predict.dtype).cuda()
        
        mols = np.array(smi_list, dtype=str)
        mols = np.expand_dims(mols, 1)
        matrix1 = np.repeat(mols, len(mols), 1)
        matrix2 = np.repeat(np.transpose(mols), len(mols), 0)
        mol_duplicate_matrix = matrix1==matrix2
        mol_duplicate_matrix = 1*mol_duplicate_matrix
        mol_duplicate_matrix = torch.tensor(mol_duplicate_matrix, dtype=ba_predict.dtype).cuda()

        
        

        onehot_labels = torch.eye(bsz).cuda()
        indicater_matrix = pocket_duplicate_matrix + mol_duplicate_matrix - 2*onehot_labels
        
        #print(ba_predict.shape)
        ba_predict = ba_predict *  self.logit_scale.exp().detach()
        ba_predict = indicater_matrix * -1e6 + ba_predict

        return ba_predict, self.logit_scale.exp(), loss_diffusion #_pocket, ba_predict_mol, loss_diffusion

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates











class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x




@register_model_architecture("drugclip", "drugclip")
def drugclip_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)



