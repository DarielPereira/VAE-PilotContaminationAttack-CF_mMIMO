import numpy as np
import numpy.linalg as alg
import math

def ComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p):
    """
    Compute uplink SE for attack-agnostic MMSE combining in a cell-free massive MIMO network.

    INPUT>
    :param Hhat: L*N x nbrOfRealizations x K estimated channels (attack-agnostic)
    :param H: L*N x nbrOfRealizations x K true channels (may include attacker effect)
    :param D: L x K AP-UE assignment matrix
    :param C: N x N x L x K channel estimation error correlation matrices
    :param tau_c: coherence block length
    :param tau_p: pilot length
    :param nbrOfRealizations: number of channel realizations
    :param N: number of antennas per AP
    :param K: number of UEs
    :param L: number of APs
    :param p: uplink transmit power per UE (mW)

    OUTPUT>
    :return SE_MMSE: Kx1 vector of uplink SE for each UE
    """

    eyeN = np.identity(N)
    prelogFactor = (tau_c - tau_p) / tau_c

    SE_MMSE = np.zeros((K, 1), dtype=complex)

    # Loop over channel realizations
    for n in range(nbrOfRealizations):

        # Loop over all UEs
        for k in range(K):
            servingAPs, = np.where(D[:, k] == 1)
            La = len(servingAPs)

            if La == 0:
                continue

            # Build true and estimated channels for the APs serving UE k
            Hall = np.zeros((N*La, K), dtype=complex)
            Hhatall = np.zeros((N*La, K), dtype=complex)
            C_tot = np.zeros((N*La, N*La), dtype=complex)

            for idx, l in enumerate(servingAPs):
                Hall[idx*N:(idx+1)*N, :] = H[l*N:(l+1)*N, n, :]
                Hhatall[idx*N:(idx+1)*N, :] = Hhat[l*N:(l+1)*N, n, :]
                C_tot[idx*N:(idx+1)*N, idx*N:(idx+1)*N] = np.sum(C[:, :, l, :], axis=2)

            # Compute attack-agnostic MMSE combining vector
            v = p * alg.inv(p * (Hhatall @ Hhatall.conj().T) + p * C_tot + np.identity(N*La)) @ Hhatall[:, k]

            # Compute SINR using **true channels** (including attack)
            signal = p * np.abs(v.conj().T @ Hall[:, k])**2
            interference = p * np.sum([np.abs(v.conj().T @ Hall[:, j])**2 for j in range(K) if j != k])
            noise = v.conj().T @ v

            SINR = signal / (interference + noise)

            # Update SE
            SE_MMSE[k] += prelogFactor * np.log2(1 + SINR) / nbrOfRealizations

    return np.sum(SE_MMSE), SE_MMSE