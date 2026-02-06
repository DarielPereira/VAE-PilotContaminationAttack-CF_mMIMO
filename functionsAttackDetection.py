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
    Calculates the probability of being under attack based on the clean distribution.

    Logic:
    - Clean users have HIGH KL divergence (centered around mu).
    - Attacked users have LOW KL divergence (left tail).
    - We calculate the Cumulative Distribution Function (CDF).
    - A very low CDF value implies the score is extremely unlikely to be from the clean distribution's left side.
    - We map this to attack probability.

    We use a Sigmoid function centered at (mu - 3*sigma) to provide a calibrated [0, 1] probability.
    Why mu - 3*sigma? Because statistically, 99.7% of clean data is above this.
    Crossing this threshold implies high confidence of anomaly.

    :param scores: Array of scores to evaluate
    :param mu: Mean of clean distribution
    :param sigma: Std Dev of clean distribution
    :return: Array of probabilities [0, 1]
    """

    # Threshold for 50% probability (Shifted 3 sigmas to the left)
    # This ensures that "normal" clean samples (around mu) have near 0 probability of attack.
    threshold = mu - 2 * sigma

    # Scale factor for the sigmoid steepness.
    # Adjusted so the transition is smooth but decisive around the threshold.
    scale = sigma

    # Sigmoid function: 1 / (1 + exp( (x - threshold) / scale ))
    # If x is much lower than threshold (Attack), exponent is negative large, exp is small, Prob -> 1.
    # If x is much higher than threshold (Clean), exponent is positive large, exp is large, Prob -> 0.

    # We use (x - threshold) because low scores = attack.
    # Positive exponent makes denominator large -> Prob 0 (Correct for clean/high scores)

    # Calculate Z-score distance from the "safety threshold"
    z_dist = (scores - threshold) / (scale * 0.5)  # 0.5 factor to sharpen the transition slightly

    probabilities = 1 / (1 + np.exp(z_dist))

    return probabilities


