"""
This script contains functions for pilot assignment and AP cooperation clustering strategies
"""


import numpy as np

from functionsUtils import db2pow


def PilotAssignment(gainOverNoisedB, tau_p, K, mode):
    """Compute the pilot assignment for a set of UEs
    INPUT>
    :param gainOverNoisedB: matrix with dimensions (L, K) containing the channel gains
    :param tau_p: number of pilots
    :param K: number of UEs
    :param mode: pilot assignment mode
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # to store pilot assignment
    pilotIndex = -1 * np.ones((K), int)

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'DCC':

            # Determine the pilot assignment
            for k in range(0, K):

                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                if k <= tau_p - 1:  # Assign orthogonal pilots to the first tau_p UEs
                    pilotIndex[k] = k

                else:  # Assign pilot for remaining users

                    # Compute received power to the master AP from each pilot
                    pilotInterference = np.zeros(tau_p)

                    for t in range(tau_p):
                        pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master, :k][pilotIndex[:k] == t]))

                    # Find the pilot with least received power
                    bestPilot = np.argmin(pilotInterference)
                    pilotIndex[k] = bestPilot

    return pilotIndex

def AP_Assignment(gainOverNoisedB, tau_p, K, L, pilotIndex, mode):
    """Compute AP cooperation cluster assignments
    INPUT>
    :param gainOverNoisedB: matrix with dimensions (L, K) containing the channel gains
    :param tau_p: number of pilots
    :param K: number of UEs
    :param L: number of APs
    :param pilotIndex: pilot assignment
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """
    # to store AP assignment
    D = np.zeros((L, K))

    match mode:
        case 'DCC':
            # Each AP serves the UE with the strongest channel condition on each of the pilots
            for l in range(L):
                for t in range(tau_p):
                    pilotUEs, = np.where(pilotIndex == t)
                    if len(pilotUEs) > 0:
                        UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
                        D[l, pilotUEs[UEindex]] = 1

        case 'ALL':
            D = np.ones((L, K))

    return D









