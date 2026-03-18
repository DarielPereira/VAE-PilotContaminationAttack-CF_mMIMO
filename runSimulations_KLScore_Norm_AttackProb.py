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
from functionsDataProcessing import complex_to_real_batch
from functionsGraphs import plot_histograms, plot_roc_curve, plot_scatter, plot_attack_probability, plot_shapedKL_histogram
from functionsAttackDetection import compute_link_scores_vectorized
from functionsUtils import drawingSetup
from functionsAttackDetection import fit_clean_distribution, calculate_attack_probability

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


##Setting Parameters
configuration = {
    'nbrOfSetups': 500,            # number of communication network setups
    'nbrOfRealizations': 100,      # number of channel realizations per sample
    'L': [36, 49, 64],                       # number of APs
    'N': 2,                       # number of antennas per AP
    'K': [20, 30],                       # number of UEs
    'T': None,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 100,                  # length of the coherence block
    'tau_p': None,                   # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'p_attacker': 5,            # uplink transmit power per attacker in mW
    'n_attackers': [40, 60],             # Number of attackers in the system. Use [1,1] for single-adversary
    'cell_side': 250,             # side of the square cell in m
    'ASD_varphi': math.radians(10), # Azimuth angle - Angular Standard Deviation in the local scattering model
    'Testing': False              # if True, fix random seed for reproducibility
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
K = configuration['K']
tau_c = configuration['tau_c']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['Testing']
p_attacker = configuration['p_attacker']

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
all_pilot_labels = []

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    print(f'Generating setup {setup_iter + 1}/{nbrOfSetups} with {K} connected UEs......')
    # get the number of APs, UEs and attackers for this setup
    L = np.random.choice(configuration['L'])
    k_min = configuration['K'][0]
    k_max = configuration['K'][1]
    K = np.random.randint(k_min, k_max+1)
    # Set pilot length equal to K for orthogonal pilots if not specified in configuration
    tau_p = K if configuration['tau_p'] is None else configuration['tau_p']
    # Set T equal to L for no clustering if not specified in configuration
    T = L if configuration['T'] is None else configuration['T']
    n_attackers_min = configuration['n_attackers'][0]
    n_attackers_max = configuration['n_attackers'][1]
    n_attackers = np.random.randint(n_attackers_min, n_attackers_max+1)

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
    current_pilot_labels = []
    for t in range(tau_p):
        if t in attacked_pilots:
            current_pilot_labels.append(1) # User under attack
        else:
            current_pilot_labels.append(0) # Clean user

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
    all_pilot_labels.extend(current_pilot_labels)

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
all_pilot_labels = np.array(all_pilot_labels)
all_avg_frob_PsiInv_emp = np.array(all_avg_frob_PsiInv_emp)

# From the visualizations below, only those used for publishing have been left uncommented.
# The rest can be easily re-enabled by uncommenting the corresponding blocks.

# # --- HISTOGRAM VISUALIZATION ---
# # --- PART 1: LINK LEVEL PLOTS ---
# plot_histograms(all_scores_total, all_labels,
#                 x_label_str='Joint Anomaly Score Distribution\n(Reconstruction + beta * KL)',
#             y_label_str='Density', filename='./Graphs/hist_link_JointAnomalyScore.pdf')
#
# plot_histograms(all_scores_recon, all_labels,
#                 x_label_str='Reconstruction Error (MSE) Distribution',
#             y_label_str='Density', filename='./Graphs/hist_link_ReconstructionAnomalyScore.pdf')
# #
# plot_histograms(all_scores_kl, all_labels,
#                 x_label_str='KL Divergence Score Distribution',
#             y_label_str='Density', filename='./Graphs/hist_link_KLAnomalyScore.pdf')
#
# plot_histograms(all_frob_Diff_PsiInv_emp_PsiInv_Bth, all_labels,
#                 x_label_str=r"$|| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} - \mathbf{\Psi}_{l,t_k}^{\textsuperscript{th}} ||$",
#             y_label_str='Density', filename='./Graphs/hist_link_frobDiff.pdf')
# #
# plot_histograms(all_frob_PsiInv_emp, all_labels,
#                 x_label_str=r"$|| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} ||$",
#             y_label_str='Density', filename='./Graphs/hist_link_frob.pdf')
# #
# # --- PART 2: USER LEVEL PLOTS (Averages) ---
# plot_histograms(all_avg_total, all_pilot_labels,
#                 x_label_str='Joint Anomaly Score Distribution\n(Reconstruction + beta * KL)',
#             y_label_str='Density', filename='./Graphs/hist_pilot_JointAnomalyScore.pdf')
#
# plot_histograms(all_avg_recon, all_pilot_labels,
#                 x_label_str='Reconstruction Error (MSE) Distribution',
#             y_label_str='Density', filename='./Graphs/hist_pilot_ReconstructionAnomalyScore.pdf')
# #
# plot_histograms(all_avg_kl, all_pilot_labels,
#                 x_label_str='KL Divergence Score Distribution',
#             y_label_str='Density', filename='./Graphs/hist_pilot_KLAnomalyScore.pdf')
#
# plot_histograms(all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth, all_pilot_labels,
#                x_label_str=r"$\frac{1}{L}\sum_{l=1}^{L}\big|\big| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} - \mathbf{\Psi}_{l,t_k}^{\textsuperscript{th}} \big|\big|$",
#             y_label_str='Density', filename='./Graphs/hist_pilot_frobDiff.pdf')
#
# # Frob. norm histogram (Figure PPT-Low power vs high power)
# plot_histograms(all_avg_frob_PsiInv_emp, all_pilot_labels,
#                 x_label_str=r"$\frac{\sum_{l=1}^{L}\big|\big| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} \big|\big|}{L}$",
#             y_label_str='Density', filename='./Graphs/hist_pilot_frob.pdf')
#
#
#
# # --- SCATTER PLOTS VISUALIZATION ---
# # Link Level
# # 2D plot: X = Frobenius Norm of the Difference between PsiInv_emp and PsiInv_th, Y = KL Divergence
# plot_scatter(all_frob_Diff_PsiInv_emp_PsiInv_Bth, all_scores_kl, all_labels,
#              x_label_str=r"$\big|\big| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} - \mathbf{\Psi}_{l,t_k}^{\textsuperscript{th}} \big|\big|$",
#              y_label_str=r'KL divergence',
#              filename='./Graphs/scatter_link_frob_Diff_vs_kl.pdf')
#
# # Pilot Level (Figure 5 PAPER)
# # 2D plot: X = Average Frobenius Norm of the Difference between PsiInv_emp and PsiInv_th, Y = KL Divergence
# plot_scatter(all_avg_frob_Diff_PsiInv_emp_PsiInv_Bth, all_avg_kl, all_pilot_labels,
#              x_label_str=r"$\frac{1}{L}\sum_{l=1}^{L}\big|\big| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} - \mathbf{\Psi}_{l,t_k}^{\textsuperscript{th}} \big|\big|$",
#              y_label_str=r"Anomaly score $s_{t_k}$",
#              filename='./Graphs/scatter_pilot_frob_Diff_vs_kl.pdf')
#
# # Link Level
# # 2D plot: X = Frobenius Norm of PsiInv_emp, Y = KL Divergence
# plot_scatter(all_frob_PsiInv_emp, all_scores_kl, all_labels,
#              x_label_str=r"$|| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} ||$",
#              y_label_str=r'KL divergence',
#              filename='./Graphs/scatter_link_frob_PsiInv.pdf')
#
# # Pilot Level
# # 2D plot: X = Average Frobenius Norm of PsiInv, Y = Average KL Divergence
# plot_scatter(all_avg_frob_PsiInv_emp, all_avg_kl, all_pilot_labels,
#              x_label_str=r"$\frac{\sum_{l=1}^{L}\big|\big| \mathbf{\Psi}_{l,t_k}^{\textsuperscript{emp}} \big|\big|}{L}$",
#              y_label_str=r"Anomaly score $s_{t_k}$",
#              filename='./Graphs/scatter_pilot_frob_PsiInv.pdf')
#
#
# # --- PROBABILITY ANALYSIS (PILOT LEVEL KL) --- (Figure 4 PAPER)
# probs,_,_ = plot_attack_probability(all_avg_kl, all_pilot_labels)

# # Plot ROC curve for KL-based detection at the pilot level
# plot_roc_curve(all_pilot_labels, probs)

# --- SHAPED KL DIVERGENCE SCORES HISTOGRAM --- (Figure 3 PAPER)
# plot_shapedKL_histogram(all_avg_kl, all_pilot_labels,
#                 x_label_str=r"Anomaly score $s_{t_k}$",
#             y_label_str='Density', filename='./Graphs/hist_pilot_Shaped_KLAnomalyScore.pdf')
