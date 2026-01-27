"""
This code is intended to simulate a pilot contamination attack in a cell-free massive MIMO network.
"""
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import ComputeSE_uplink
from functionsComputeNMSE_uplink import ComputeNMSE_uplink
from functionsAttack import generateAttack

import math
import numpy as np


##Setting Parameters
configuration = {
    'nbrOfSetups': 1,            # number of communication network setups
    'nbrOfRealizations': 3,      # number of channel realizations per sample
    'L': 4,                     # number of APs
    'N': 2,                       # number of antennas per AP
    'K': 3,                    # number of UEs
    'T': 4,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                 # length of the coherence block
    'tau_p': 3,                  # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'p_attacker': 10,            # uplink transmit power of the attacker in mW
    'cell_side': 200,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'Testing': True               # if True, fix random seed for reproducibility
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

    # 4. Set Vector of power distribution for the attacker
    dict_attack = generateAttack(L, N, tau_p, cell_side, ASD_varphi, p_attacker,
                   APpositions, bool_testing=bool_testing, attack_mode='uniform')

    # 5. Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B_th, C_th, B_emp = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p, dict_attack)



    # 6. Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    system_SE, UEs_SE = ComputeSE_uplink(Hhat, H, D, C_th, tau_c, tau_p,
                               nbrOfRealizations, N, K, L, p)

    # 7. Compute NMSE for all the UEs
    system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot \
        = ComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex, dict_attack)


    print(f'Sum Spectral Efficiency: {system_SE}')
    print(f'System NMSE: {system_NMSE}')

