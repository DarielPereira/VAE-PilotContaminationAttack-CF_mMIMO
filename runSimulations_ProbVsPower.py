"""
This code is intended to simulate a pilot contamination attack in a cell-free massive MIMO network.
"""
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment

from functionsComputeNMSE_uplink import ComputeNMSE_uplink
from functionsAttack import generateAttack
from functionsChannelEstimates import channelEstimates
from functionsAttackDetection import (attack_detection_scores, calculate_attack_probability,
                                      calculate_attack_probability_upper_tail)
from functionscVAE import VAEModel
from functionsGraphs import plot_attack_probability_generic, plot_crossentropy_vs_power
from scipy.stats import norm

from functionsGraphs import plot_nmse_cdfs


import math
import numpy as np
import torch
import matplotlib.pyplot as plt


configuration = {
    'nbrOfSetups': 300,            # number of communication network setups
    'nbrOfRealizations': 100,      # number of channel realizations per sample
    'L': [36, 49, 64],                       # number of APs
    'N': 2,                       # number of antennas per AP
    'K': [20, 30],                       # number of UEs
    'T': None,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 100,                  # length of the coherence block
    'tau_p': None,                   # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                           # uplink transmit power per UE in mW
    'p_attackers': [5, 25, 50, 75, 100, 125, 150, 175],            # uplink transmit power per attacker in mW
    'n_attackers': [40, 60],             # Number of attackers in the system
    'cell_side': 250,             # side of the square cell in m
    'ASD_varphi': math.radians(10), # Azimuth angle - Angular Standard Deviation in the local scattering model
    'Testing': False              # if True, fix random seed for reproducibility
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
# L = configuration['L']
N = configuration['N']
# K = configuration['K']
# T = configuration['T']
tau_c = configuration['tau_c']
# tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['Testing']
p_attackers = configuration['p_attackers']
n_attackers = configuration['n_attackers']  # Extract number of attackers

# To store results
crossEntropies_VAE = np.zeros((len(p_attackers)))
crossEntropies_Norm = np.zeros((len(p_attackers)))
crossEntropies_random = np.zeros((len(p_attackers)))
crossEntropies_optimal = np.zeros((len(p_attackers)))

for p_idx, p_attacker in enumerate(p_attackers):
    print('p_attacker: ', p_attacker)

    all_pilot_labels = []
    all_VAE_scores = []
    all_norm_scores = []

    # Run over all the setups
    for setup_iter in range(nbrOfSetups):

        # get the number of APs, UEs and attackers for this setup
        L = np.random.choice(configuration['L'])
        k_min = configuration['K'][0]
        k_max = configuration['K'][1]
        K = np.random.randint(k_min, k_max + 1)
        # Set pilot length equal to K for orthogonal pilots if not specified in configuration
        tau_p = K if configuration['tau_p'] is None else configuration['tau_p']
        # Set T equal to L for no clustering if not specified in configuration
        T = L if configuration['T'] is None else configuration['T']
        n_attackers_min = configuration['n_attackers'][0]
        n_attackers_max = configuration['n_attackers'][1]
        n_attackers = np.random.randint(n_attackers_min, n_attackers_max + 1)

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
                                     APpositions, n_attackers=n_attackers, bool_testing=bool_testing,
                                     attack_mode='random_selective')

        # Get labels for each link based on attacked pilots
        attacked_pilots = dict_attack['pilot_indices']

        # Labeling for Users (One label per user)
        current_pilot_labels = []
        for t in range(tau_p):
            if t in attacked_pilots:
                current_pilot_labels.append(1)  # User under attack
            else:
                current_pilot_labels.append(0)  # Clean user
        all_pilot_labels.extend(current_pilot_labels)

        # 5. Generate channel realizations with estimates and estimation error matrices
        # dict_attack now contains info for multiple attackers, handled inside channelEstimates
        Hhat, H, _B_th, C_th, _B_emp, PsiInv_th, PsiInv_emp = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p,
                                                                               pilotIndex, p,
                                                                               dict_attack, bool_testing=bool_testing)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = VAEModel(input_dim=(2 * PsiInv_th.shape[0]) ** 2, latent_dim=6, hidden_dims=[16, 8])
        # Ensure this path matches your trained model file
        model.load_model(f'./Models/cVAE_model_NbrSamples_1228330_ASD_10_P_100_Normalized_PsiInv_Large.pth', device)

        model.to(device)
        model.eval()


        # 6. Run attack detection pipeline
        VAE_scores = attack_detection_scores(model, PsiInv_emp, current_pilot_labels, device=device, attack_algorithm='VAE')
        Norm_scores = attack_detection_scores(model, PsiInv_emp, current_pilot_labels, device=device,
                                              attack_algorithm='Norm')

        all_VAE_scores.extend(VAE_scores)
        all_norm_scores.extend(Norm_scores)


    all_pilot_labels = np.array(all_pilot_labels)
    all_VAE_scores = np.array(all_VAE_scores)
    all_norm_scores = np.array(all_norm_scores)

    # Boolean masks for clean and attacked pilots
    clean_indices = (all_pilot_labels == 0)
    clean_VAE_scores = all_VAE_scores[clean_indices]
    clean_Norm_scores = all_norm_scores[clean_indices]

    # Fit Gaussian (Mean and Std)
    mu_clean_VAE, sigma_clean_VAE = norm.fit(clean_VAE_scores)
    mu_clean_Norm, sigma_clean_norm = norm.fit(clean_Norm_scores)

    VAE_probabilities = calculate_attack_probability(all_VAE_scores, mu_clean_VAE, sigma_clean_VAE)
    Norm_probabilities = calculate_attack_probability_upper_tail(all_norm_scores, mu_clean_Norm, sigma_clean_norm)
    random_probabilities = np.random.rand(all_pilot_labels.shape[0])  # Random probabilities for baseline

    # Calculate cross-entropy for each method
    crossEntropies_VAE[p_idx] = -np.mean(all_pilot_labels * np.log(VAE_probabilities + 1e-12)
                                       + (1 - all_pilot_labels) * np.log(1 - VAE_probabilities + 1e-12))
    crossEntropies_Norm[p_idx] = -np.mean(all_pilot_labels * np.log(Norm_probabilities + 1e-12)
    + (1 - all_pilot_labels) * np.log(1 - Norm_probabilities + 1e-12))
    crossEntropies_random[p_idx] = -np.mean(all_pilot_labels * np.log(random_probabilities + 1e-12)
                                          + (1 - all_pilot_labels) * np.log(1 - random_probabilities + 1e-12))
    crossEntropies_optimal[p_idx] = -np.mean(all_pilot_labels * np.log(all_pilot_labels + 1e-12)
                                           + (1 - all_pilot_labels) * np.log(1 - all_pilot_labels + 1e-12))  # Optimal (perfect) probabilities




#
# # Plot hist of the probabilities for attacked vs non-attacked users
# plot_attack_probability_generic(np.array(VAE_probabilities), all_pilot_labels, x_label_str=r'scores',
#                                 save_path='./Graph_Tests/VAE_Prob_Distribution.png')
# plot_attack_probability_generic(np.array(Norm_probabilities), all_pilot_labels, x_label_str=r'Norms',
#                                 save_path='./Graph_Tests/Norm_Prob_Distribution.png')

# Plot attacker power vs cross-entropy for the four methods
plot_crossentropy_vs_power(p_attackers, crossEntropies_VAE, crossEntropies_Norm,
                           crossEntropies_random, crossEntropies_optimal)
