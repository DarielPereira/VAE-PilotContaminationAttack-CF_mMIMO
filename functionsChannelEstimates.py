"""
This function generates the channel realizations and estimations of these channels for all the UEs in the entire network.
"""

import numpy
import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random


def channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p, dict_attack=None, bool_testing=True):
    """Generate the channel realizations and estimations of these channels for all the UEs in the entire network.
    The channels are assumed to be correlated Rayleigh fading and the MMSE estimator is used.
    INPUT>
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel between
    UE k and AP l (normalized by noise variance)
    :param nbrOfRealizations: number of channel realizations
    :param L: number of Aps
    :param K: number of UEs
    :param N: number of antennas per AP
    :param tau_p: number of orthogonal pilots
    :param pilotIndex: vector containing the pilot assigned to each UE
    :param p: Uplink transmit power per UE (same for everyone)

    OUTPUT>
    Hhat: matrix with dimensions L*N x nbrOfRealizations x K where (:, n, k) is the estimated collective channel to
                    UE k in the channel realization n
    H: matrix with dimensions L*N x nbrOfRealizations x K where (:, n, k) is the true realization of the collective
                    channel to UE k in the channel realization n.
    B_th: matrix with dimensions N x N x L x K where (:,:,l,k) is the theoretical spatial correlation matrix of the channel estimate
                    between AP l and UE k (normalized by noise variance)
    C_th: matrix with dimension N x N x L x K where (:,:,l,k) is the theoretical spatial correlation matrix of the channel estimation
                    error between AP l and UE k (normalized by noise variance)
    B_emp: matrix with dimensions N x N x L x K where (:,:,l,k) is the empirical spatial correlation matrix of the channel estimate
                    between AP l and UE k (normalized by noise variance)
    """

    if bool_testing:
        np.random.seed(0)

    # If there is an attacker, extract its parameters
    if dict_attack is not None:
        R_attack = dict_attack['R']  # (N,N,L,n_attackers)
        p_attack = dict_attack['p_attack']  # (tau_p,1)
        pilot_indices_attack = dict_attack['pilot_indices']
        n_attackers = dict_attack.get('n_attackers', 1)

    # Generate uncorrelated Rayleigh fading channel realizations for users
    H = np.random.randn(L * N, nbrOfRealizations, K) + 1j * np.random.randn(L * N, nbrOfRealizations, K)

    # Go through all channels and apply the spatial correlation matrices
    for l in range(L):
        for k in range(K):
            # Apply correlation to the uncorrelated channel realizations
            Rsqrt = spalg.sqrtm(R[:, :, l, k])  # square root of the correlation matrix
            H[l * N:(l + 1) * N, :, k] = np.sqrt(0.5) * Rsqrt @ H[l * N:(l + 1) * N, :, k]

    # If there is an attacker, generate its channel realizations and apply the correlation matrices
    if dict_attack is not None:
        # H_attacker shape: (L*N, nbrOfRealizations, n_attackers)
        H_attacker = np.random.randn(L * N, nbrOfRealizations, n_attackers) + 1j * np.random.randn(L * N,
                                                                                                   nbrOfRealizations,
                                                                                                   n_attackers)

        for i in range(n_attackers):
            for l in range(L):
                # Apply correlation to the uncorrelated channel realizations of attacker i
                # R_attack is now (N, N, L, n_attackers)
                Rsqrt_attack = spalg.sqrtm(R_attack[:, :, l, i])  # square root of the correlation matrix
                H_attacker[l * N:(l + 1) * N, :, i] = np.sqrt(0.5) * Rsqrt_attack @ H_attacker[l * N:(l + 1) * N, :, i]

    # Perform channel estimation
    # Store identity matrix of size NxN
    eyeN = np.identity(N)

    # Generate realizations of normalized noise
    Np = np.sqrt(0.5) * (
                np.random.randn(N, nbrOfRealizations, L, tau_p) + 1j * np.random.randn(N, nbrOfRealizations, L, tau_p))

    # Prepare to store results
    Hhat = np.zeros((L * N, nbrOfRealizations, K), dtype=complex)
    B_th = np.zeros((R.shape), dtype=complex)
    C_th = np.zeros((R.shape), dtype=complex)

    # Go through all the APs
    for l in range(L):

        # Go through all the pilots
        for t in range(tau_p):

            # Compute processed pilot signal for all the UEs that use pilot t with an additional scaling factor
            # \sqrt(tau_p)
            yp = np.sqrt(p) * tau_p * np.sum(H[l * N: (l + 1) * N, :, t == pilotIndex], axis=2) + np.sqrt(tau_p) * Np[
                :, :, l, t]

            # If there is an attacker and it uses pilot t, add its contribution to the processed pilot signal
            if dict_attack is not None and t in pilot_indices_attack:
                # Sum contribution from ALL attackers attacking pilot t
                for i in range(n_attackers):
                    yp += np.sqrt(p_attack[t, 0]) * tau_p * H_attacker[l * N:(l + 1) * N, :, i]

            # Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p * tau_p * np.sum(R[:, :, l, t == pilotIndex], axis=2) + eyeN)

            # # # Uncomment to make the channel estimator aware of the attacker
            # if dict_attack is not None and t in pilot_indices_attack:
            #     for i in range(n_attackers):
            #         PsiInv += p_attack[t, 0] * tau_p * R_attack[:, :, l, i]

            # PsiInv = np.zeros((N, N), dtype=complex)
            # # Compute the Psi empirical covariance matrix from the channel realizations
            # for k in range(K):
            #     PsiInv = (yp @ yp.conj().T) / nbrOfRealizations

            # Go through all the UEs that use pilot t
            pilotsharingUEs, = np.where(t == pilotIndex)
            if len(pilotsharingUEs) > 0:
                for k in pilotsharingUEs:
                    # Compute the MSE estimate
                    RPsi = R[:, :, l, k] @ alg.inv(PsiInv)
                    Hhat[l * N: (l + 1) * N, :, k] = np.sqrt(p) * RPsi @ yp

                    # Compute the spatial correlation matrix of the estimation
                    B_th[:, :, l, k] = p * tau_p * RPsi @ R[:, :, l, k]

                    # Compute the spatial correlation matrix of the estimation error
                    C_th[:, :, l, k] = R[:, :, l, k] - B_th[:, :, l, k]

    # B_emp covariance matriz estimation from channel realizations
    # This matrix captures the actual covariance of the channel estimates
    # and will be used as input of the cVAE model during inference.
    B_emp = np.zeros((N, N, L, K), dtype=complex)
    for l in range(L):
        for k in range(K):
            Hhat_lk = Hhat[l * N:(l + 1) * N, :, k]  # N × Nreal
            B_emp[:, :, l, k] = (
                                        Hhat_lk @ Hhat_lk.conj().T
                                ) / nbrOfRealizations

    return Hhat, H, B_th, C_th, B_emp
