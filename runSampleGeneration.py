"""
This code is intended to generate the samples that will be used for training and testing the cVAE.
Samples will be composed of (B,R) pairs, where B is the covariance matrix of the channel estimate and R is the channel correlation matrix.
The samples are generated for a cell-free massive MIMO system free of attacks.
"""
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment
from functionsAttack import generateAttack
from functionsDataProcessing import Dataset_cVAE
import numpy.linalg as alg
from tqdm import tqdm

import math
import numpy as np


##Setting Parameters
configuration = {
    'nbrOfSetups': 2500,            # number of communication network setups
    'L': 9,                     # number of APs
    'N': 2,                       # number of antennas per AP
    'K': 5,                    # number of UEs
    'T': None,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                 # length of the coherence block
    'tau_p': None,                  # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 200,                     # uplink transmit power per UE in mW
    'cell_side': 200,            # side of the square cell in m
    'ASD_varphi': math.radians(5),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'bool_Testing': False               # if True, fix random seed for reproducibility
}

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']

N = configuration['N']
L = configuration['L']
K = configuration['K']
T = L if configuration['T'] is None else configuration['T']         # set no clustering by default
tau_c = configuration['tau_c']
tau_p = K if configuration['tau_p'] is None else configuration['tau_p']         # set pilot length equal to K for orthogonal pilots
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['bool_Testing']

# Create dataset object
dataset = Dataset_cVAE()

# Run over all the setups
for setup_iter in tqdm(range(nbrOfSetups), desc="Generating Setups", unit="setup"):

    # 1. Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
        generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=bool_testing, seed=setup_iter))

    # 2. Compute pilot assignment
    pilotIndex = PilotAssignment(gainOverNoisedB, tau_p, K, mode='DCC')

    # 3. Set AP cooperation cluster assignment
    D = AP_Assignment(gainOverNoisedB, tau_p, K, L, pilotIndex, mode='DCC')

    B = np.zeros((R.shape), dtype=complex)

    # Store identity matrix of size NxN
    eyeN = np.identity(N)

    # Go through all the APs
    for l in range(L):
        # Go through all the pilots
        for t in range(tau_p):
            # Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p * tau_p * np.sum(R[:, :, l, t == pilotIndex], axis=2) + eyeN)

            # Go through all the UEs that use pilot t
            pilotsharingUEs, = np.where(t == pilotIndex)
            if len(pilotsharingUEs) > 0:
                for k in pilotsharingUEs:
                    # Compute the MSE estimate
                    RPsi = R[:, :, l, k] @ alg.inv(PsiInv)

                    # Compute the spatial correlation matrix of the estimation
                    B[:, :, l, k] = p * tau_p * RPsi @ R[:, :, l, k]


    dataset.add_from_simulation(B, R)

# # Normalize B data in dataset
dataset.normalize_B()

# Transform into suitable format for cVAE training (real-valued samples)
dataset.to_real_representation()

dataset.metadata = {
    'N': N,
    'K': K,
    'tau_c': tau_c,
    'tau_p': tau_p,
    'p': p,
    'cell_side': cell_side,
    'ASD_varphi': ASD_varphi
    }

# Save dataset to file
dataset.save(f'./TrainingData/cVAE_dataset_N_{N}_NbrSamples_{dataset.__len__()}_ASD_{ASD_varphi}_P_{p}_Normalized.npz')


