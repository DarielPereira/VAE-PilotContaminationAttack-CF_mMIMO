"""
Dataset class for CF-mMIMO channel estimation statistics
Attack-agnostic training (only clean samples)
"""

import numpy as np
import pickle
import os
import torch

class Dataset_cVAE:
    def __init__(self, B_shape=None, R_shape=None, metadata=None):
        """
        Dataset storing (B, R) pairs:
        - B: covariance of channel estimate
        - R: channel correlation matrix (conditioning variable)

        Shapes are optional but recommended for consistency checks.
        """
        self.B_samples = []
        self.R_samples = []

        self.B_shape = B_shape
        self.R_shape = R_shape
        self.metadata = metadata if metadata is not None else {}

    # -------------------------------------------------
    # Basic dataset operations
    # -------------------------------------------------
    def add_sample(self, B, R):
        """
        Add a single (B, R) sample.

        :param B: np.array, covariance of estimation
        :param R: np.array, channel correlation matrix
        """
        if self.B_shape is not None:
            assert B.shape == self.B_shape, f"B shape {B.shape} != {self.B_shape}"
        if self.R_shape is not None:
            assert R.shape == self.R_shape, f"R shape {R.shape} != {self.R_shape}"

        self.B_samples.append(B)
        self.R_samples.append(R)

    def __len__(self):
        return len(self.B_samples)

    def get_sample(self, idx):
        """
        Return a single sample (B, R)
        """
        return self.B_samples[idx], self.R_samples[idx]

    def sample_batch(self, batch_size):
        """
        Random batch sampling (without replacement)

        :return:
            B_batch: (batch_size, *B_shape)
            R_batch: (batch_size, *R_shape)
        """
        assert batch_size <= len(self), "Batch size larger than dataset"

        idx = np.random.choice(len(self), batch_size, replace=False)
        B_batch = np.array([self.B_samples[i] for i in idx])
        R_batch = np.array([self.R_samples[i] for i in idx])

        return B_batch, R_batch

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    def save(self, filepath):
        """
        Save dataset to disk
        """
        data = {
            'B_samples': self.B_samples,
            'R_samples': self.R_samples,
            'B_shape': self.B_shape,
            'R_shape': self.R_shape,
            'metadata': self.metadata
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[Dataset] Saved {len(self)} samples to {filepath}")

    def load(self, filepath):
        """
        Load dataset from disk
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.B_samples = data['B_samples']
        self.R_samples = data['R_samples']
        self.B_shape = data['B_shape']
        self.R_shape = data['R_shape']

        print(f"[Dataset] Loaded {len(self)} samples from {filepath}")

    # -------------------------------------------------
    # Dataset generation from simulation outputs
    # -------------------------------------------------
    def add_from_simulation(self, B, R, D=None):
        """
        Add multiple samples from simulation outputs.

        Typical use:
        - B: (N, N, L, K)
        - R: (N, N, L, K)

        Each (l, k) pair becomes one sample.
        Optionally filtered by D (AP-UE association).
        """
        N, _, L, T = B.shape

        for l in range(L):
            for t in range(T):

                if D is not None:
                    if D[l, t] == 0:
                        continue

                self.add_sample(
                    B=B[:, :, l, t],
                    R=R[:, :, l, t]
                )

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def normalize_PsiInv(self, eps=1e-12):
        """
        Normalize B samples (trace normalization).
        Useful for stable VAE training.
        """
        for i in range(len(self.B_samples)):
            tr = np.trace(self.B_samples[i]).real
            self.B_samples[i] = self.B_samples[i] / (tr + eps)

    def to_real_representation(self):
        """
        Convert complex matrices to real-valued representations:
        [Re(.)  Im(.);
         -Im(.) Re(.)]

        Useful if using real-valued neural networks.
        """
        def complex_to_real(M):
            return np.block([
                [np.real(M),  np.imag(M)],
                [-np.imag(M), np.real(M)]
            ])

        self.B_samples = [complex_to_real(B) for B in self.B_samples]
        self.R_samples = [complex_to_real(R) for R in self.R_samples]

        self.B_shape = self.B_samples[0].shape
        self.R_shape = self.R_samples[0].shape

def complex_to_real_batch(B_emp):
    """
    Transform a batch of complex matrices to real-valued representations.

    B_emp: np.ndarray of shape (N, N, L, K)
    Returns: torch.Tensor of shape (L*K, 2N*2N)
    """
    N, _, L, K = B_emp.shape
    B_real_list = []
    for l in range(L):
        for k in range(K):
            B_cplx = B_emp[:, :, l, k]
            B_real = np.block([
                [np.real(B_cplx), np.imag(B_cplx)],
                [-np.imag(B_cplx), np.real(B_cplx)]
            ])
            B_real_list.append(B_real.flatten())
    B_real_tensor = torch.tensor(np.stack(B_real_list), dtype=torch.float32)
    return B_real_tensor  # shape (L*K, (2N)^2)