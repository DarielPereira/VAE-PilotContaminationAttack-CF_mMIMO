"""
This code is intended to simulate a pilot contamination attack in a cell-free massive MIMO network.
"""
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import ComputeSE_uplink
from functionsComputeNMSE_uplink import ComputeNMSE_uplink
from functionsAttack import generateAttack
from functionscVAE import VAEModel
from functionsAttack import complex_to_real_batch, compute_link_scores_vectorized, plot_histograms
from functionsUtils import drawingSetup
from functionsAttackDetection import fit_clean_distribution, calculate_attack_probability, plot_roc_curve

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


##Setting Parameters
configuration = {
    'nbrOfSetups': 500,            # number of communication network setups
    'nbrOfRealizations': 100,      # number of channel realizations per sample
    'L': 9,                       # number of APs
    'N': 2,                       # number of antennas per AP
    'K': 6,                       # number of UEs
    'T': 9,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                  # length of the coherence block
    'tau_p': 6,                   # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'p_attacker': 100,            # uplink transmit power per attacker in mW
    'n_attackers': 20,             # Number of attackers in the system
    'cell_side': 100,             # side of the square cell in m
    'ASD_varphi': math.radians(10), # Azimuth angle - Angular Standard Deviation in the local scattering model
    'Testing': False              # if True, fix random seed for reproducibility
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
K = configuration['K']
T = configuration['T']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['Testing']
p_attacker = configuration['p_attacker']
n_attackers = configuration['n_attackers']  # Extract number of attackers

# To store results (Link Level)
all_scores_joint = []
all_scores_recon = []
all_scores_kl = []
all_frob_Diff_PsiInv_emp_PsiInv_Bth = []
all_frob_PsiInv_emp = []
all_labels = []  # 0: Clean, 1: Attacked

# To store results (User Level - Averaged)
all_avg_joint = []
all_avg_recon = []
all_avg_kl = []
all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth = []
all_avg_frob_PsiInv_emp = []
all_user_labels = []

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    print(f'Generating setup {setup_iter + 1}/{nbrOfSetups} with {K} connected UEs......')

    # 1. Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
        generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=bool_testing, seed=setup_iter))

    # 2. Compute pilot assignment
    pilotIndex = PilotAssignment(gainOverNoisedB, tau_p, K, mode='DCC')

    # 3. Set AP cooperation cluster assignment
    D = AP_Assignment(gainOverNoisedB, tau_p, K, L, pilotIndex, mode='DCC')

    # 4. Set Vector of power distribution for the attacker(s)
    # UPDATED: Passing n_attackers to the function
    dict_attack = generateAttack(L, N, tau_p, cell_side, ASD_varphi, p_attacker,
                   APpositions, n_attackers=n_attackers, bool_testing=bool_testing, attack_mode='random_selective')

    # 5. Generate channel realizations with estimates and estimation error matrices
    # dict_attack now contains info for multiple attackers, handled inside channelEstimates
    Hhat, H, _B_th, C_th, _B_emp, PsiInv_th, PsiInv_emp = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p,
                                                  dict_attack, bool_testing=bool_testing)

    # --- STORE FROB NORMS ---
    for l in range(L):
        for t in range(tau_p):
            frob_norm = np.linalg.norm(PsiInv_emp[:, :, l, t], 'fro')
            all_frob_PsiInv_emp.append(frob_norm)
    for t in range(tau_p):
        avg_frob_norm = np.mean([np.linalg.norm(PsiInv_emp[:, :, l, t], 'fro') for l in range(L)])
        all_avg_frob_PsiInv_emp.append(avg_frob_norm)

    # --- NORMALIZATION STEP ---
    # Normalize B_emp AND B_th using trace normalization to ensure Frobenius comparison is valid
    for l in range(L):
        for t in range(tau_p):
            # Normalize Empirical
            tr = np.trace(PsiInv_emp[:,:, l, t]).real
            PsiInv_emp[:, :, l, t] = PsiInv_emp[:, :, l, t]/(tr + 1e-12)

            # Normalize Theoretical (Required for valid Frobenius difference)
            tr_th = np.trace(PsiInv_th[:,:, l, t]).real
            PsiInv_th[:, :, l, t] = PsiInv_th[:, :, l, t]/(tr_th + 1e-12)
    # --------------------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VAEModel(input_dim=(2*PsiInv_th.shape[0])**2, latent_dim=6, hidden_dims=[16, 8])
    # Ensure this path matches your trained model file
    model.load_model(f'./Models/cVAE_model_NbrSamples_112500_ASD_5_P_200_Normalized_PsiInv.pth', device)

    model.to(device)
    model.eval()

    beta = 0.2  # weight for KL in the score

    # Compute the score for each UE-AP link and the averages per user
    (joint_scores, recon_scores, kl_scores, frob_Diff_Bemp_Bth,
     avg_joint, avg_recon, avg_kl, avg_frob_Diff_Bemp_Bth) = compute_link_scores_vectorized(
        model, PsiInv_emp, PsiInv_th, device=device, beta=beta
    )

    # Flatten link scores
    joint_scores_flat = joint_scores.T.flatten()
    recon_scores_flat = recon_scores.T.flatten()
    kl_scores_flat = kl_scores.T.flatten()
    frob_Diff_Bemp_flat = frob_Diff_Bemp_Bth.T.flatten()

    # Get labels for each link based on attacked pilots
    attacked_pilots = dict_attack['pilot_indices']

    # Labeling for Links (Iterate APs then UEs to match flatten order of scores.T)
    current_labels = []
    for l in range(L):
        for t in range(tau_p):
            # A link is considered attacked if the pilot of the UE is among the attacked pilots
            if t in attacked_pilots:
                current_labels.append(1)  # Atacado
            else:
                current_labels.append(0)  # Clean

    # Labeling for Users (One label per user)
    current_user_labels = []
    for t in range(tau_p):
        if t in attacked_pilots:
            current_user_labels.append(1) # User under attack
        else:
            current_user_labels.append(0) # Clean user

    # 8. Save results of current setup
    # Link Level
    all_scores_joint.extend(joint_scores_flat)
    all_scores_recon.extend(recon_scores_flat)
    all_scores_kl.extend(kl_scores_flat)
    all_frob_Diff_PsiInv_emp_PsiInv_Bth.extend(frob_Diff_Bemp_flat)
    all_labels.extend(current_labels)

    # User Level
    all_avg_joint.extend(avg_joint)
    all_avg_recon.extend(avg_recon)
    all_avg_kl.extend(avg_kl)
    all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth.extend(avg_frob_Diff_Bemp_Bth)
    all_user_labels.extend(current_user_labels)

# Transform into numpy arrays for further processing
all_scores_total = np.array(all_scores_joint)
all_scores_recon = np.array(all_scores_recon)
all_scores_kl = np.array(all_scores_kl)
all_frob_Diff_PsiInv_emp_PsiInv_Bth = np.array(all_frob_Diff_PsiInv_emp_PsiInv_Bth)
all_labels = np.array(all_labels)
all_frob_PsiInv_emp = np.array(all_frob_PsiInv_emp)

# Transform user averages into numpy arrays
all_avg_total = np.array(all_avg_joint)
all_avg_recon = np.array(all_avg_recon)
all_avg_kl = np.array(all_avg_kl)
all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth = np.array(all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth)
all_user_labels = np.array(all_user_labels)
all_avg_frob_PsiInv_emp = np.array(all_avg_frob_PsiInv_emp)

# --- HISTOGRAM VISUALIZATION ---
# Pass both link-level and user-level data including Frobenius norms
plot_histograms(
    all_scores_total, all_scores_recon, all_scores_kl, all_frob_Diff_PsiInv_emp_PsiInv_Bth, all_labels,
    all_avg_total, all_avg_recon, all_avg_kl, all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth, all_user_labels
)

# --- SCATTER PLOT VISUALIZATION (LINK LEVEL) ---
# 2D plot: X = Frobenius Norm of the Difference between B_emp and B_th, Y = KL Divergence
plt.figure(figsize=(10, 6))

# Identify clean and attacked indices
clean_idx = (all_labels == 0)
attacked_idx = (all_labels == 1)

# Plot attacked samples (first, to be in background if crowded, or flip order if preferred)
plt.scatter(all_frob_Diff_PsiInv_emp_PsiInv_Bth[attacked_idx], all_scores_kl[attacked_idx],
            color='crimson', alpha=0.5, label='Attacked Links', s=10)

# Plot clean samples
plt.scatter(all_frob_Diff_PsiInv_emp_PsiInv_Bth[clean_idx], all_scores_kl[clean_idx],
            color='dodgerblue', alpha=0.5, label='Clean Links', s=10)

plt.xlabel('Frobenius Norm (Empirical vs Theoretical)')
plt.ylabel('KL Divergence')
plt.title('2D Analysis (Link Level): Frobenius Norm of the Difference between B_emp and B_th vs KL Divergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
filename_scatter = 'scatter_link_frob_Diff_vs_kl.png'
plt.savefig(filename_scatter)
print(f"Saved {filename_scatter}")

plt.show()
plt.close()

# --- SCATTER PLOT VISUALIZATION (USER LEVEL) ---
# 2D plot: X = Average Frobenius Norm of the Difference between B_emp and B_th, Y = Average KL Divergence
plt.figure(figsize=(10, 6))

# Identify clean and attacked indices for users
clean_user_idx = (all_user_labels == 0)
attacked_user_idx = (all_user_labels == 1)

# Plot attacked users
plt.scatter(all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth[attacked_user_idx], all_avg_kl[attacked_user_idx],
            color='orange', alpha=0.5, label='Attacked Users', s=20)

# Plot clean users
plt.scatter(all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth[clean_user_idx], all_avg_kl[clean_user_idx],
            color='teal', alpha=0.5, label='Clean Users', s=20)

plt.xlabel('Average Frobenius Norm Score (Empirical vs Theoretical)')
plt.ylabel('Average KL Divergence')
plt.title('2D Analysis (User Level): Avg Frobenius Norm of the Difference between B_emp and B_th vs Avg KL Divergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
filename_user_scatter = 'scatter_user_frob_Diff_vs_kl.png'
plt.savefig(filename_user_scatter)
print(f"Saved {filename_user_scatter}")

plt.show()
plt.close()

# --- SCATTER PLOT VISUALIZATION (LINK LEVEL) ---
# 2D plot: X = Frobenius Norm of B_emp, Y = KL Divergence
plt.figure(figsize=(10, 6))

# Identify clean and attacked indices
clean_idx = (all_labels == 0)
attacked_idx = (all_labels == 1)

# Plot attacked samples (first, to be in background if crowded, or flip order if preferred)
plt.scatter(all_frob_PsiInv_emp[attacked_idx], all_scores_kl[attacked_idx],
            color='crimson', alpha=0.5, label='Attacked Links', s=10)

# Plot clean samples
plt.scatter(all_frob_PsiInv_emp[clean_idx], all_scores_kl[clean_idx],
            color='dodgerblue', alpha=0.5, label='Clean Links', s=10)

plt.xlabel('Frobenius Norm of B_emp')
plt.ylabel('KL Divergence')
plt.title('2D Analysis (Link Level): Frobenius Norm of B_emp vs KL Divergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
filename_scatter = 'scatter_link_frob_Bemp_vs_kl.png'
plt.savefig(filename_scatter)
print(f"Saved {filename_scatter}")

plt.show()
plt.close()

# --- SCATTER PLOT VISUALIZATION (USER LEVEL) ---
# 2D plot: X = Average Frobenius Norm of B_emp, Y = Average KL Divergence
plt.figure(figsize=(10, 6))

# Identify clean and attacked indices for users
clean_user_idx = (all_user_labels == 0)
attacked_user_idx = (all_user_labels == 1)

# Plot attacked users
plt.scatter(all_avg_frob_PsiInv_emp[attacked_user_idx], all_avg_kl[attacked_user_idx],
            color='orange', alpha=0.5, label='Attacked Users', s=20)

# Plot clean users
plt.scatter(all_avg_frob_PsiInv_emp[clean_user_idx], all_avg_kl[clean_user_idx],
            color='teal', alpha=0.5, label='Clean Users', s=20)

plt.xlabel('Average Frobenius of B_emp')
plt.ylabel('Average KL Divergence')
plt.title('2D Analysis (User Level): Avg Frobenius Norm of B_emp vs Avg KL Divergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
filename_user_scatter = 'scatter_user_frob_Bemp_vs_kl.png'
plt.savefig(filename_user_scatter)
print(f"Saved {filename_user_scatter}")

plt.show()
plt.close()



# --- PROBABILITY ANALYSIS (USER LEVEL KL) ---
print("\n--- ATTACK PROBABILITY ANALYSIS ---")

# 1. Fit distribution to Clean Data Generated in Simulation (Self-Reference)
# Extraemos los scores KL de los usuarios que sabemos que son limpios (label 0)
clean_kl_scores = all_avg_kl[clean_user_idx]

if len(clean_kl_scores) > 0:
    # Usamos estos datos para ajustar la Gaussiana del detector
    mu_clean, sigma_clean = fit_clean_distribution(clean_kl_scores)
    print(f"Clean Distribution (KL) Fitted from Simulation Data: Mean={mu_clean:.4f}, Std={sigma_clean:.4f}")

    # 2. Calculate Probabilities for ALL Simulated Users (Clean & Attacked)
    # El detector evalúa qué tan probable es que cada muestra pertenezca a la distribución limpia
    probs = calculate_attack_probability(all_avg_kl, mu_clean, sigma_clean)

    # 3. Plot Probability Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probs[clean_user_idx], color='teal', alpha=0.6, label='Clean Users', bins=50)
    plt.hist(probs[attacked_user_idx], color='orange', alpha=0.6, label='Attacked Users', bins=50)
    plt.xlabel('Calculated Probability of Attack')
    plt.ylabel('Count')
    plt.title('Attack Probability Distribution (Based on Avg KL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./Graph_Tests/hist_attack_probability.png')
    print("Saved hist_attack_probability.png")
    plt.show()
    plt.close()

    # 4. Plot ROC Curve
    plot_roc_curve(all_user_labels, probs, title="ROC Curve: KL-Based Detection")

else:
    print("Error: No clean user samples generated in simulation to fit the detector.")