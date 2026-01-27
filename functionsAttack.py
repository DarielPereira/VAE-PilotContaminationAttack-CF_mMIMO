"""
This script contains functions to generate and characterize a pilot contamination attacker
in a cell-free massive MIMO network.
"""

import numpy as np
import math
import os

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