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
    'nbrOfSetups': 1000,            # number of communication network setups
    'L': [36, 49, 64],                     # number of APs
    'N': 2,                       # number of antennas per AP
    'K': [20, 30],                    # number of UEs
    'T': None,                       # number of APs connected to each CPU (set equal to L for no clustering)
    'tau_c': 20,                 # length of the coherence block
    'tau_p': None,                  # length of the pilot sequences (set equal to K for orthogonal pilots)
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 400,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'bool_Testing': False               # if True, fix random seed for reproducibility
}

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']

N = configuration['N']
tau_c = configuration['tau_c']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
bool_testing = configuration['bool_Testing']

# Create dataset object
dataset = Dataset_cVAE()

# Run over all the setups
for setup_iter in tqdm(range(nbrOfSetups), desc="Generating Setups", unit="setup"):
    # get the number of APs and UEs for this setup
    L = np.random.choice(configuration['L'])
    k_min = configuration['K'][0]
    k_max = configuration['K'][1]
    K = np.random.randint(k_min, k_max+1)
    # Set pilot length equal to K for orthogonal pilots if not specified in configuration
    tau_p = K if configuration['tau_p'] is None else configuration['tau_p']
    # Set T equal to L for no clustering if not specified in configuration
    T = L if configuration['T'] is None else configuration['T']

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

    # Preallocate matrices
    PsiInv_th = np.zeros((N, N, L, tau_p), dtype=complex)

    # Go through all the APs
    for l in range(L):
        # Go through all the pilots
        for t in range(tau_p):

            # Compute the matrix that is inverted in the MMSE estimator
            temp_PsiInv_th = (p * tau_p * np.sum(R[:, :, l, t == pilotIndex], axis=2) + eyeN)

            PsiInv_th[:, :, l, t] = temp_PsiInv_th


    dataset.add_from_simulation(PsiInv_th, R)

# # Normalize B data in dataset
dataset.normalize_PsiInv()

# Transform into suitable format for cVAE training (real-valued samples)
dataset.to_real_representation()

dataset.metadata = {
    'tau_c': tau_c,
    'tau_p': tau_p,
    'p': p,
    'cell_side': cell_side,
    'ASD_varphi': ASD_varphi
    }

# Save dataset to file
dataset.save(f'./TrainingData/cVAE_dataset_N_{N}_NbrSamples_{dataset.__len__()}_ASD_{int(180*ASD_varphi/np.pi)}_P_{p}_Normalized_PsiInv_Large.npz')


