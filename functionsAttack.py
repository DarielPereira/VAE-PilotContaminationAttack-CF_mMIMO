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
                   n_attackers=1,
                   bool_testing=True,
                   attack_mode='uniform'):
    """
    Generate the characteristics of pilot contamination attackers.

    INPUT>
    :param L: number of APs
    :param N: number of antennas per AP
    :param tau_p: number of pilots
    :param cell_side: side of the square coverage area (in meters)
    :param ASD_varphi: angular standard deviation (radians) for local scattering
    :param p_attacker: total uplink transmit power of the attacker (mW) (per attacker)
    :param APpositions: (L x 1) complex array with AP positions
    :param n_attackers: Number of attackers to generate
    :param bool_testing: if True, fix random seed
    :param attack_mode: pilot power allocation strategy
                        ('uniform', 'single', 'random')

    OUTPUT>
    :attack: dictionary containing all attacker characteristics
             'R' will now be of shape (N, N, L, n_attackers)
    """

    if bool_testing:
        np.random.seed(1)


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
    # 1. Generate attackers positions (Randomly distributed)
    # -------------------------------------------------
    # Shape: (n_attackers,)
    attackPositions = (np.random.rand(n_attackers) + 1j * np.random.rand(n_attackers)) * cell_side

    # -------------------------------------------------
    # 2. Distances and large-scale fading for each attacker
    # -------------------------------------------------
    # APpositions shape: (L, 1)
    # attackPositions shape: (n_attackers,) -> reshape to (1, n_attackers) for broadcasting
    # Resulting distances shape: (L, n_attackers)
    distances = np.sqrt(
        distanceVertical**2 +
        np.abs(APpositions - attackPositions.reshape(1, -1))**2
    )

    gainOverNoisedB_attack = (
        constantTerm
        - alpha * np.log10(distances)
        - noiseVariancedBm
    ) # Shape: (L, n_attackers)

    # -------------------------------------------------
    # 3. Spatial correlation matrices for each attacker
    # -------------------------------------------------
    # R_attack shape: (N, N, L, n_attackers)
    R_attack = np.zeros((N, N, L, n_attackers), dtype=complex)

    for i in range(n_attackers):
        for l in range(L):
            angle_varphi = np.angle(attackPositions[i] - APpositions[l])
            R_attack[:, :, l, i] = (
                db2pow(gainOverNoisedB_attack[l, i]) *
                localScatteringR(N, angle_varphi, ASD_varphi, antennaSpacing)
            )

    # -------------------------------------------------
    # 4. Pilot power allocation (Shared strategy)
    # -------------------------------------------------
    # We assume all attackers use the same strategy and pilot allocation
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
            num_selected = np.random.randint(1, tau_p)
            selected_indices = np.random.choice(tau_p, num_selected, replace=False)
            p_attack[selected_indices] = p_attacker/num_selected

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
        'positions': attackPositions,
        'p_attack': p_attack,
        'pilot_indices': pilot_indices,
        'gainOverNoisedB': gainOverNoisedB_attack,
        'R': R_attack,
        'mode': attack_mode,
        'n_attackers': n_attackers
    }

    return attack






