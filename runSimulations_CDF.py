"""
This code is intended to simulate a pilot contamination attack in a cell-free massive MIMO network.
"""
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment

from functionsComputeNMSE_uplink import ComputeNMSE_uplink
from functionsAttack import generateAttack

from functionsGraphs import plot_nmse_cdfs


import math
import numpy as np
import torch
import matplotlib.pyplot as plt


##Setting Parameters
configuration = {
    'nbrOfSetups': 500,            # number of communication network setups
    'nbrOfRealizations': 100,      # number of channel realizations per sample
    'L': 36,                       # number of APs
    'N': 2,                       # number of antennas per AP
    'K': 20,                       # number of UEs
    'T': 36,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                  # length of the coherence block
    'tau_p': 20,                   # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'p_attacker_single': 200,            # uplink transmit power total in mW for single attacker scenario
    'p_attacker_multiple': 100,            # uplink transmit power total in mW for multiple attackers scenario
    'n_attackers': 20,             # Number of attackers in the system
    'cell_side': 400,             # side of the square cell in m
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
p_attacker_single = configuration['p_attacker_single']
p_attacker_multiple = configuration['p_attacker_multiple']
n_attackers = configuration['n_attackers']  # Extract number of attackers

# To store results (Link Level)
users_NMSEs_NoAttack = np.zeros((nbrOfSetups*K))
users_NMSEs_SingleAttacker = np.zeros((nbrOfSetups*K))
users_NMSEs_MultiAttacker = np.zeros((nbrOfSetups*K))

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
    dict_single_attack = generateAttack(L, N, tau_p, cell_side, ASD_varphi, p_attacker_single,
                   APpositions, n_attackers=1, bool_testing=bool_testing, attack_mode='random')

    # 4. Set Vector of power distribution for the attacker(s)
    dict_multi_attack = generateAttack(L, N, tau_p, cell_side, ASD_varphi, p_attacker_multiple/n_attackers,
                                        APpositions, n_attackers=20, bool_testing=bool_testing, attack_mode='random')

    dict_multi_attack['pilot_indices'] = dict_single_attack['pilot_indices']
    dict_multi_attack['p_attack'] = dict_single_attack['p_attack']*(p_attacker_multiple/p_attacker_single)/n_attackers

    # 5. Compute NMSE for all the UEs
    _, UEs_NMSEs_noAttack, _, _ \
        = ComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

    # 5. Compute NMSE for all the UEs
    _, UEs_NMSEs_single, _, _ \
        = ComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex, dict_single_attack)

    # 5. Compute NMSE for all the UEs
    _, UEs_NMSEs_multiple, _, _ \
        = ComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex, dict_multi_attack)

    users_NMSEs_NoAttack[setup_iter * K: (setup_iter + 1) * K] = UEs_NMSEs_noAttack[:]
    users_NMSEs_SingleAttacker[setup_iter * K: (setup_iter + 1) * K] = UEs_NMSEs_single[:]
    users_NMSEs_MultiAttacker[setup_iter * K: (setup_iter + 1) * K] = UEs_NMSEs_multiple[:]

# Plot the CDFs of the NMSEs for the different scenarios
plot_nmse_cdfs(users_NMSEs_NoAttack, users_NMSEs_SingleAttacker, users_NMSEs_MultiAttacker)