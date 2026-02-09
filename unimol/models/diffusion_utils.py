import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean 

# auxiliary function for diffusion
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

def q_v_pred(log_v0, t, log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v, batch):
    # compute q(vt | v0)
    log_cumprod_alpha_t = extract(log_alphas_cumprod_v, t, batch)
    log_1_min_cumprod_alpha = extract(log_one_minus_alphas_cumprod_v, t, batch)

    log_probs = log_add_exp(
        log_v0 + log_cumprod_alpha_t,
        log_1_min_cumprod_alpha - np.log(30)
    )
    return log_probs

def q_v_sample(log_v0, log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v, t, batch):
        log_qvt_v0 = q_v_pred(log_v0, t, log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, 30)
        return sample_index, log_sample

def sample_time(num_graphs, num_timesteps, device, method='symmetric', low_ratio=0.3):
    if method == 'symmetric':
        time_step = torch.randint(
            0, num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / num_timesteps
        return time_step, pt
    elif method == 'mid_symmetric':
        # only use mid timesteps, i.e., [300,700]
        t_min = int(num_timesteps * low_ratio)
        t_max = num_timesteps - t_min 
        time_step = torch.randint(
            t_min, t_max, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / (t_max - t_min)
        return time_step, pt

    else:
        raise ValueError


def compute_v_Lt(log_v_model_prob, log_v0, log_v_true_prob, t, batch):
    kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
    decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
    assert kl_v.shape == decoder_nll_v.shape
    mask = (t == 0).float()[batch]
    loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
    return loss_v


# atom type diffusion process
def q_v_pred_one_timestep(log_vt_1, t, log_alphas_v, log_one_minus_alphas_v, batch):
    # q(vt | vt-1)
    log_alpha_t = extract(log_alphas_v, t, batch)
    log_1_min_alpha_t = extract(log_one_minus_alphas_v, t, batch)

    # alpha_t * vt + (1 - alpha_t) 1 / K
    log_probs = log_add_exp(
        log_vt_1 + log_alpha_t,
        log_1_min_alpha_t - np.log(30)
    )
    return log_probs

 # atom type generative process
def q_v_posterior(log_v0, log_vt, log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v, log_alphas_v, log_one_minus_alphas_v, t, batch):
    # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
    t_minus_1 = t - 1
    # Remove negative values, will not be used anyway for final decoder
    t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
    log_qvt1_v0 = q_v_pred(log_v0, t_minus_1,  log_alphas_cumprod_v, log_one_minus_alphas_cumprod_v, batch)
    unnormed_logprobs = log_qvt1_v0 + q_v_pred_one_timestep(log_vt, t, log_alphas_v, log_one_minus_alphas_v, batch)
    log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
    return log_vt1_given_vt_v0

def diffusion_loss(
    ligand_pos,           
    ligand_pos_perturbed, 
    pred_ligand_pos,     
    pred_ligand_v,      
    log_ligand_v0,       
    log_ligand_vt,    

    log_alphas_cumprod_v, 
    log_one_minus_alphas_cumprod_v, 
    log_alphas_v, 
    log_one_minus_alphas_v,
    
    time_step,        
    batch_ligand 
):
    ## using FP32 to prevent NAN/INF and then cast back to FP16
    target_pos = ligand_pos.float()
    pred_pos   = pred_ligand_pos.float()

    per_atom_pos_mse = ((pred_pos - target_pos) ** 2).sum(-1)
    loss_pos = scatter_mean(per_atom_pos_mse, batch_ligand, dim=0).mean()

    logits_v = pred_ligand_v.float()
    log_v_recon = F.log_softmax(logits_v, dim=-1)

    log_v0 = log_ligand_v0.float()
    log_vt = log_ligand_vt.float()

    log_v_model_prob = q_v_posterior(
        log_v_recon,
        log_vt,
        log_alphas_cumprod_v.float(),
        log_one_minus_alphas_cumprod_v.float(),
        log_alphas_v.float(),
        log_one_minus_alphas_v.float(),
        time_step,
        batch_ligand,
    )
    log_v_true_prob = q_v_posterior(
        log_v0,
        log_vt,
        log_alphas_cumprod_v.float(),
        log_one_minus_alphas_cumprod_v.float(),
        log_alphas_v.float(),
        log_one_minus_alphas_v.float(),
        time_step,
        batch_ligand,
    )

    kl_v = compute_v_Lt(
        log_v_model_prob=log_v_model_prob,
        log_v0=log_v0,                      
        log_v_true_prob=log_v_true_prob,
        t=time_step,
        batch=batch_ligand,
    )
    loss_v = kl_v.mean()

    loss = loss_pos + 100 * loss_v

    return (loss.to(pred_ligand_pos.dtype), loss_pos.to(pred_ligand_pos.dtype), loss_v.to(pred_ligand_pos.dtype))