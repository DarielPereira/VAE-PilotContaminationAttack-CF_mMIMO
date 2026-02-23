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
    'nbrOfSetups': 300,            # number of communication network setups
    'nbrOfRealizations': 100,      # number of channel realizations per sample
    'L': [36, 49, 64],                       # number of APs
    'N': 2,                       # number of antennas per AP
    'K': [20, 30],                       # number of UEs
    'T': None,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                  # length of the coherence block
    'tau_p': None,                   # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'p_attacker_single': 250,            # uplink transmit power total in mW for single attacker scenario
    'n_attackers': [40, 60],            # Number of attackers in the system
    'cell_side': 250,             # side of the square cell in m
    'ASD_varphi': math.radians(10), # Azimuth angle - Angular Standard Deviation in the local scattering model
    'Testing': False              # if True, fix random seed for reproducibility
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
N = configuration['N']
T = configuration['T']
tau_c = configuration['tau_c']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['Testing']
p_attacker_single = configuration['p_attacker_single']

# To store results (Link Level)
users_NMSEs_NoAttack = []
users_NMSEs_SingleAttacker = []
users_NMSEs_MultiAttacker = []

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
    p_attacker_multiple = 5 * n_attackers  # Total power for multiple attackers scenario
                                           # (assuming each attacker has 5 mW, adjust as needed)

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

    users_NMSEs_NoAttack = users_NMSEs_NoAttack + UEs_NMSEs_noAttack[:].flatten().tolist()
    users_NMSEs_SingleAttacker = users_NMSEs_SingleAttacker + UEs_NMSEs_single[:].flatten().tolist()
    users_NMSEs_MultiAttacker = users_NMSEs_MultiAttacker + UEs_NMSEs_multiple[:].flatten().tolist()

# Plot the CDFs of the NMSEs for the different scenarios
plot_nmse_cdfs(np.array(users_NMSEs_NoAttack), np.array(users_NMSEs_SingleAttacker),
               np.array(users_NMSEs_MultiAttacker))