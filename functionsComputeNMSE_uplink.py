"""
This function computes the NMSE for every user in a cell-free massive MIMO system.
"""

import numpy as np
import numpy.linalg as alg


def ComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex, dict_attack=None):
    """ Compute the NMSE for every user with optional pilot contamination attacker """

    p = 100
    eyeN = np.identity(N)
    Psi = np.zeros((N, N, L, tau_p), dtype=complex)
    UEs_NMSE = np.zeros(K)

    for t in range(tau_p):
        pilotSharing_UEs, = np.where(pilotIndex == t)

        for l in range(L):
            # start with the sum of UEs' contributions
            Psi_term = sum([tau_p * p * R[:, :, l, k] for k in pilotSharing_UEs])

            # add attacker contribution if it affects this pilot
            if dict_attack is not None and t in dict_attack['pilot_indices']:
                Psi_term += tau_p * dict_attack['p_attack'][t, 0] * dict_attack['R'][:, :, l]

            Psi[:, :, l, t] = alg.inv(eyeN + Psi_term)

    for k in range(K):
        if sum(D[:, k]) > 0:
            t_k = pilotIndex[k]
            serving_APs, = np.where(D[:, k] == 1)

            UEs_NMSE[k] = 1 - (sum([tau_p * p * np.trace(R[:, :, l, k] @ Psi[:, :, l, t_k] @ R[:, :, l, k])
                                    for l in serving_APs]) /
                               sum([np.trace(R[:, :, l, k]) for l in serving_APs])).real
        else:
            UEs_NMSE[k] = 0

    system_NMSE = np.sum(UEs_NMSE)
    worst_userXpilot = np.zeros(tau_p)
    best_userXpilot = np.zeros(tau_p)

    for t in range(tau_p):
        NMSE_pilotSharing_UEs = UEs_NMSE[pilotIndex[:] == t]
        if len(NMSE_pilotSharing_UEs) > 0:
            worst_userXpilot[t] = max(NMSE_pilotSharing_UEs)
            best_userXpilot[t] = min(NMSE_pilotSharing_UEs)

    return system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot