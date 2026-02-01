"""
This script contains functions to generate and characterize a pilot contamination attacker
in a cell-free massive MIMO network.
"""

import numpy as np
import math
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Avoid OpenMP issues
os.environ['OMP_NUM_THREADS'] = '1'

from functionsUtils import db2pow, localScatteringR


def generateAttack(L, N, tau_p, cell_side, ASD_varphi,
                   p_attacker,
                   APpositions,
                   bool_testing=True,
                   attack_mode='uniform'):
    """
    Generate the characteristics of a pilot contamination attacker.

    INPUT>
    :param L: number of APs
    :param N: number of antennas per AP
    :param tau_p: number of pilots
    :param cell_side: side of the square coverage area (in meters)
    :param ASD_varphi: angular standard deviation (radians) for local scattering
    :param p_attacker: total uplink transmit power of the attacker (mW)
    :param APpositions: (L x 1) complex array with AP positions
    :param bool_testing: if True, fix random seed
    :param attack_mode: pilot power allocation strategy
                        ('uniform', 'single', 'random')

    OUTPUT>
    :attack: dictionary containing all attacker characteristics
    """

    if bool_testing:
        np.random.seed(0)


    # -------------------------------------------------
    # System parameters (same as generateSetup)
    # -------------------------------------------------
    B = 20 * 10**6                  # bandwidth (Hz)
    noiseFigure = 7                 # dB
    noiseVariancedBm = -174 + 10 * np.log10(B) + noiseFigure

    alpha = 36.7                    # path loss exponent
    constantTerm = -30.5

    distanceVertical = 10           # AP-UE height difference (m)
    antennaSpacing = 0.5            # half-wavelength spacing

    # -------------------------------------------------
    # 1. Generate attacker position
    # -------------------------------------------------
    attackPosition = (np.random.rand() + 1j * np.random.rand()) * cell_side

    # -------------------------------------------------
    # 2. Distances and large-scale fading
    # -------------------------------------------------
    distances = np.sqrt(
        distanceVertical**2 +
        np.abs(APpositions - attackPosition)**2
    )[:, 0]

    gainOverNoisedB_attack = (
        constantTerm
        - alpha * np.log10(distances)
        - noiseVariancedBm
    ).reshape(L, 1)

    # -------------------------------------------------
    # 3. Spatial correlation matrices
    # -------------------------------------------------
    R_attack = np.zeros((N, N, L), dtype=complex)

    for l in range(L):
        angle_varphi = np.angle(attackPosition - APpositions[l])
        R_attack[:, :, l] = (
            db2pow(gainOverNoisedB_attack[l, 0]) *
            localScatteringR(N, angle_varphi, ASD_varphi, antennaSpacing)
        )

    # -------------------------------------------------
    # 4. Pilot power allocation
    # -------------------------------------------------
    p_attack = np.zeros((tau_p, 1))

    match attack_mode:

        case 'uniform':
            p_attack[:] = p_attacker / tau_p

        case 'single':
            pilot_idx = np.random.randint(0, tau_p)
            p_attack[pilot_idx] = p_attacker

        case 'random':
            temp = np.random.rand(tau_p, 1)
            p_attack = p_attacker * temp / np.sum(temp)

        case 'random_selective':
            # Randomly select a subset of pilots to attack
            num_selected = np.random.randint(1, tau_p/2 + 1)
            selected_indices = np.random.choice(tau_p, num_selected, replace=False)
            temp = np.random.rand(num_selected, 1)
            p_attack[selected_indices] = p_attacker * temp / np.sum(temp)

        case _:
            raise ValueError("Unknown attack_mode")

    # -------------------------------------------------
    # 4b. Indices of attacked pilots
    # -------------------------------------------------
    pilot_indices = np.where(p_attack[:, 0] > 0)[0]

    # -------------------------------------------------
    # 5. Pack everything into a dictionary
    # -------------------------------------------------
    attack = {
        'position': attackPosition,
        'p_attack': p_attack,
        'pilot_indices': pilot_indices,
        'gainOverNoisedB': gainOverNoisedB_attack,
        'R': R_attack,
        'mode': attack_mode
    }

    return attack


def compute_link_scores_vectorized(model, B_emp, device='cpu', beta=0.2):
    """
    Compute attack scores per UE-AP link in a vectorized manner.
    Also computes the average score per user (averaged over APs).

    Returns:
        - joint_mat: (K, L) matrix of total scores
        - recon_mat: (K, L) matrix of reconstruction errors
        - kl_mat: (K, L) matrix of KL divergences
        - avg_joint: (K,) vector of average total scores per user
        - avg_recon: (K,) vector of average reconstruction errors per user
        - avg_kl: (K,) vector of average KL divergences per user
    """
    model.eval()
    N, _, L, K = B_emp.shape

    # Transform batch of complex matrices to real flattened tensors
    x = complex_to_real_batch(B_emp).to(device)  # (L*K, 4N^2)

    with torch.no_grad():
        x_recon, mu, logvar = model(x)  # forward pass in batch
        recon_err = ((x - x_recon) ** 2).sum(dim=1)  # (L*K,)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (L*K,)
        scores = recon_err + beta * kl_div  # (L*K,)

    joint_scores_np = scores.cpu().numpy()
    recon_scores_np = recon_err.cpu().numpy()
    kl_scores_np = kl_div.cpu().numpy()

    # Reshape to (K, L) matrices
    joint_mat = joint_scores_np.reshape(L, K).T
    recon_mat = recon_scores_np.reshape(L, K).T
    kl_mat = kl_scores_np.reshape(L, K).T

    # Compute averages per user (axis 1 because shape is (K, L))
    avg_joint = np.mean(joint_mat, axis=1)
    avg_recon = np.mean(recon_mat, axis=1)
    avg_kl = np.mean(kl_mat, axis=1)

    return joint_mat, recon_mat, kl_mat, avg_joint, avg_recon, avg_kl

def complex_to_real_batch(B_emp):
    """
    Transform a batch of complex matrices to real-valued representations.

    B_emp: np.ndarray of shape (N, N, L, K)
    Returns: torch.Tensor of shape (L*K, 2N*2N)
    """
    N, _, L, K = B_emp.shape
    B_real_list = []
    for l in range(L):
        for k in range(K):
            B_cplx = B_emp[:, :, l, k]
            B_real = np.block([
                [np.real(B_cplx), np.imag(B_cplx)],
                [-np.imag(B_cplx), np.real(B_cplx)]
            ])
            B_real_list.append(B_real.flatten())
    B_real_tensor = torch.tensor(np.stack(B_real_list), dtype=torch.float32)
    return B_real_tensor  # shape (L*K, (2N)^2)


def plot_histograms(total, recon, kl, labels,
                    avg_total=None, avg_recon=None, avg_kl=None, user_labels=None):
    """
    Generates figures for the requested histograms with optimized limits.
    Handles both Link-Level (required) and User-Level (optional) data.
    Standard histograms with separated labels: Clean on top, Attacked below X axis.
    """

    def plot_single(data, lab, title, filename, color_c, color_a, is_user_level=False):
        plt.figure(figsize=(10, 6))

        # 0. Filter NaN and Inf values
        valid_mask = np.isfinite(data)
        n_dropped = len(data) - np.sum(valid_mask)

        if n_dropped > 0:
            print(f"Warning: Dropped {n_dropped} samples containing NaN or Inf values for {filename}")
            data = data[valid_mask]
            lab = lab[valid_mask]

        if len(data) == 0:
            print(f"Warning: No valid data points to plot for {filename}")
            plt.close()
            return

        # Masks based on the provided labels
        clean_mask = (lab == 0)
        attack_mask = (lab == 1)

        # 1. Determine Dynamic Limits (Discretization & Clipping)
        if len(data) > 0:
            limit_upper = np.percentile(data, 98)
            limit_lower = np.min(data)
            # Create 50 evenly spaced bins
            bins = np.linspace(limit_lower, limit_upper, 50)
        else:
            bins = 50
            limit_lower, limit_upper = 0, 1

        # Labels for legend
        l_clean = 'Clean Users' if is_user_level else 'Clean Links'
        l_attack = 'Attacked Users' if is_user_level else 'Attacked Links'

        # Dictionary to track max height to adjust ylim later
        max_height = 0

        # --- Plot histograms (Standard: Both upwards) ---
        kwargs = dict(alpha=0.6, bins=bins, density=False, histtype='stepfilled')

        # CLEAN HISTOGRAM
        if np.sum(clean_mask) > 0:
            counts_c, edges_c, _ = plt.hist(data[clean_mask], color=color_c, label=l_clean, **kwargs)
            max_height = max(max_height, max(counts_c))

            # Annotate Clean Counts (ABOVE bar)
            bin_width = edges_c[1] - edges_c[0]
            for i in range(len(counts_c)):
                if counts_c[i] > 0:
                    plt.text(edges_c[i] + bin_width / 2, counts_c[i], int(counts_c[i]),
                             ha='center', va='bottom', fontsize=8, color=color_c, fontweight='bold')

        # ATTACKED HISTOGRAM
        if np.sum(attack_mask) > 0:
            counts_a, edges_a, _ = plt.hist(data[attack_mask], color=color_a, label=l_attack, **kwargs)
            max_height = max(max_height, max(counts_a))

            # Annotate Attacked Counts (BELOW X-axis, y=0)
            bin_width = edges_a[1] - edges_a[0]
            for i in range(len(counts_a)):
                if counts_a[i] > 0:
                    # Place text just below 0 line
                    plt.text(edges_a[i] + bin_width / 2, 0, int(counts_a[i]),
                             ha='center', va='top', fontsize=8, color=color_a, fontweight='bold')

        # Draw a horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.8)

        # Limit X-axis
        plt.xlim(limit_lower, limit_upper)

        # Expand Y-axis to accommodate labels at the bottom (negative space) and top
        plt.ylim(bottom=-max_height * 0.08, top=max_height * 1.1)

        plt.title(title)
        plt.xlabel('Score Value')
        plt.ylabel('Sample Count')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        print(f"Saved {filename}")

        plt.show()
        plt.close()

    # --- PART 1: LINK LEVEL PLOTS ---
    print("\n--- Plotting Link-Level Statistics ---")
    print(f"Clean Links: {np.sum(labels == 0)}")
    print(f"Attacked Links: {np.sum(labels == 1)}")

    plot_single(total, labels,
                'Link: Total Anomaly Score Distribution\n(Reconstruction + beta * KL)',
                'hist_link_total.png', 'dodgerblue', 'crimson')

    plot_single(recon, labels,
                'Link: Reconstruction Error Distribution (MSE)',
                'hist_link_recon.png', 'green', 'orange')

    plot_single(kl, labels,
                'Link: KL Divergence Distribution',
                'hist_link_kl.png', 'purple', 'brown')

    # --- PART 2: USER LEVEL PLOTS (Averages) ---
    if avg_total is not None and user_labels is not None:
        print("\n--- Plotting User-Level (Average) Statistics ---")
        print(f"Clean Users: {np.sum(user_labels == 0)}")
        print(f"Attacked Users: {np.sum(user_labels == 1)}")

        plot_single(avg_total, user_labels,
                    'User Avg: Total Anomaly Score Distribution',
                    'hist_user_total.png', 'cornflowerblue', 'red', is_user_level=True)

        plot_single(avg_recon, user_labels,
                    'User Avg: Reconstruction Error Distribution',
                    'hist_user_recon.png', 'limegreen', 'gold', is_user_level=True)

        plot_single(avg_kl, user_labels,
                    'User Avg: KL Divergence Distribution',
                    'hist_user_kl.png', 'mediumorchid', 'saddlebrown', is_user_level=True)

    print("\nAll graphs generated and shown successfully.")
