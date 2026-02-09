"""
This script contains functions for anomaly detection based on statistical modeling
of the clean data distribution.
"""

import numpy as np
from scipy.stats import norm
import os
import torch
import matplotlib.pyplot as plt

from functionsDataProcessing import complex_to_real_batch

# Avoid OpenMP issues
os.environ['OMP_NUM_THREADS'] = '1'



def attack_detection_scores(model, PsiInv_emp, all_pilot_labels, device='cpu', beta=0.8, attack_algorithm='VAE'):
    """
    Pipeline to calculate scores, fit clean distribution, and assign probabilities.

    :param model: The VAE model (unused if algorithm is 'Norm')
    :param PsiInv_emp: Input data (N x N x L x tau_p) - Covariance matrices
    :param all_pilot_labels: Ground truth labels (0 for clean, 1 for attacked)
    :param device: 'cpu' or 'cuda'
    :param beta: hyperparameter for VAE
    :param attack_algorithm: 'VAE' or 'Norm'
    """

    all_pilot_labels = np.array(all_pilot_labels)  # Ensure it's a numpy array for indexing

    # 1. COMPUTE SCORES BASED ON ALGORITHM
    match attack_algorithm:
        case 'VAE':
            model.eval()
            N, _, L, tau_p = PsiInv_emp.shape
            PsiInv_emp_nomralized = np.zeros((PsiInv_emp.shape), dtype=PsiInv_emp.dtype)

            for l in range(L):
                for t in range(tau_p):
                    # Normalize Empirical
                    tr = np.trace(PsiInv_emp[:, :, l, t]).real
                    PsiInv_emp_nomralized[:, :, l, t] = PsiInv_emp[:, :, l, t] / (tr + 1e-12)

            # Transform batch of complex matrices to real flattened tensors
            x = complex_to_real_batch(PsiInv_emp_nomralized).to(device)  # (tau_p*L, 4N^2)

            with torch.no_grad():
                x_recon, mu, logvar = model(x)
                # Compute KL Divergence (Low KL = Anomaly/Whitening)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            scores_raw = kl_div.cpu().numpy()

            # Reshape to (L, tau_p) to aggregate over APs
            # Note: complex_to_real_batch flattens as L*tau_p.
            # We assume order is preserved: AP1_p1, AP1_p2... AP2_p1...
            # This reshape depends on how complex_to_real_batch flattens.
            # Assuming standard flattening (L, tau_p).
            scores_mat = scores_raw.reshape(L, tau_p).T

            # Aggregate: Average KL divergence across all APs for each pilot
            scores_avg = np.mean(scores_mat, axis=1)  # (tau_p,)

        case 'Norm':
            # Benchmark: Frobenius Norm of the covariance matrix
            # PsiInv_emp shape: (N, N, L, tau_p)

            # Calculate Norm over the NxN dimensions (axis 0 and 1)
            # Result shape: (L, tau_p)
            norms_mat = np.linalg.norm(PsiInv_emp, axis=(0, 1)).T

            # Aggregate: Average Norm across all APs for each pilot
            scores_avg = np.mean(norms_mat, axis=1)  # (tau_p,)

    return scores_avg


def compute_link_scores_vectorized(model, PsiInv_emp, PsiInv_th, device='cpu', beta=0.8):
    """
    Compute attack scores per UE-AP link in a vectorized manner.
    Also computes the average score per user (averaged over APs).

    Includes Frobenius Norm of (B_emp - B_th).

    Returns:
        - joint_mat: (K, L) matrix of total scores
        - recon_mat: (K, L) matrix of reconstruction errors
        - kl_mat: (K, L) matrix of KL divergences
        - frob_mat: (K, L) matrix of Frobenius norms
        - avg_joint: (K,) vector of average total scores per user
        - avg_recon: (K,) vector of average reconstruction errors per user
        - avg_kl: (K,) vector of average KL divergences per user
        - avg_frob: (K,) vector of average Frobenius norms per user
    """
    model.eval()
    N, _, L, tau_p = PsiInv_emp.shape

    # --- 1. VAE SCORES ---
    # Transform batch of complex matrices to real flattened tensors
    x = complex_to_real_batch(PsiInv_emp).to(device)  # (L*K, 4N^2)

    with torch.no_grad():
        x_recon, mu, logvar = model(x)  # forward pass in batch
        recon_err = ((x - x_recon) ** 2).sum(dim=1)  # (L*K,)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (L*K,)
        scores = recon_err - 10*kl_div  # (L*K,)

    joint_scores_np = scores.cpu().numpy()
    recon_scores_np = recon_err.cpu().numpy()
    kl_scores_np = kl_div.cpu().numpy()

    # --- 2. FROBENIUS NORM ---
    # Calculate difference
    diff = PsiInv_emp - PsiInv_th
    # Calculate Frobenius norm over the NxN dimensions (axis 0 and 1)
    # Resulting shape will be (L, K)
    frob_errors = np.linalg.norm(diff, ord='fro', axis=(0, 1))

    # --- 3. RESHAPE & AVERAGE ---
    # VAE outputs are (L*K) ordered AP-first. Reshape (L, K) then Transpose to (K, L)
    joint_mat = joint_scores_np.reshape(L, tau_p).T
    recon_mat = recon_scores_np.reshape(L, tau_p).T
    kl_mat = kl_scores_np.reshape(L, tau_p).T

    # Frobenius is already (L, K) from linalg.norm, so just Transpose to (K, L)
    frob_mat = frob_errors.T

    # Compute averages per user (axis 1 because shape is (K, L))
    avg_joint = np.mean(joint_mat, axis=1)
    avg_recon = np.mean(recon_mat, axis=1)
    avg_kl = np.mean(kl_mat, axis=1)
    avg_frob = np.mean(frob_mat, axis=1)

    return joint_mat, recon_mat, kl_mat, frob_mat, avg_joint, avg_recon, avg_kl, avg_frob


def fit_clean_distribution(clean_scores):
    """
    Fits a Gaussian distribution to the clean scores (User Averaged KL Divergence).

    :param clean_scores: Array of KL scores from the reference dataset (clean samples)
    :return: (mu, sigma) of the fitted Gaussian
    """
    mu, sigma = norm.fit(clean_scores)
    return mu, sigma


def calculate_attack_probability(scores, mu, sigma):
    """
    Calculates probability for metrics where ATTACK corresponds to LOW scores (e.g., KL Divergence).
    Focus: LEFT TAIL of the distribution.
    """
    # Threshold for 50% probability (Shifted 2.5 sigmas to the left)
    threshold = mu - 2 * sigma
    scale = sigma

    # Sigmoid function for Left Tail
    z_dist = (scores - threshold) / (scale * 0.5)
    probabilities = 1 / (1 + np.exp(z_dist))

    return probabilities


def calculate_attack_probability_upper_tail(scores, mu, sigma):
    """
    Calculates probability for metrics where ATTACK corresponds to HIGH scores (e.g., Frobenius Norm/Energy).
    Focus: RIGHT TAIL of the distribution.

    Logic:
    - Clean users are centered around mu.
    - Attacked users (high energy) are in the right tail.
    - Threshold set at mu + 2.5*sigma.
    """
    # # Normalize scores to avoid numerical issues in the exponential
    # scores = scores/np.max(scores)

    # Threshold for 50% probability (Shifted 2.5 sigmas to the right)
    threshold = mu + 2 * sigma
    scale = sigma

    # Sigmoid function for Right Tail
    # We use negative z_dist because High Score should -> Probability 1
    # If score >> threshold, exp is small negative, denom -> 1, Prob -> 1.
    z_dist = (scores - threshold) / (scale * 0.5)
    probabilities = 1 / (1 + np.exp(-z_dist))

    return probabilities



