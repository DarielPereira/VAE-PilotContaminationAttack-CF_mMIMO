"""
This script contains functions for anomaly detection based on statistical modeling
of the clean data distribution.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def fit_clean_distribution(clean_scores):
    """
    Fits a Gaussian distribution to the clean scores (User Averaged KL Divergence).

    :param clean_scores: Array of KL scores from the reference dataset (clean samples)
    :return: (mu, sigma) of the fitted Gaussian
    """
    mu, sigma = norm.fit(clean_scores)
    return mu, sigma


def calculate_attack_probability(scores, mu, sigma):
    """
    Calculates the probability of being under attack based on the clean distribution.

    Logic:
    - Clean users have HIGH KL divergence (centered around mu).
    - Attacked users have LOW KL divergence (left tail).
    - We calculate the Cumulative Distribution Function (CDF).
    - A very low CDF value implies the score is extremely unlikely to be from the clean distribution's left side.
    - We map this to attack probability.

    We use a Sigmoid function centered at (mu - 3*sigma) to provide a calibrated [0, 1] probability.
    Why mu - 3*sigma? Because statistically, 99.7% of clean data is above this.
    Crossing this threshold implies high confidence of anomaly.

    :param scores: Array of scores to evaluate
    :param mu: Mean of clean distribution
    :param sigma: Std Dev of clean distribution
    :return: Array of probabilities [0, 1]
    """

    # Threshold for 50% probability (Shifted 3 sigmas to the left)
    # This ensures that "normal" clean samples (around mu) have near 0 probability of attack.
    threshold = mu - 2.5 * sigma

    # Scale factor for the sigmoid steepness.
    # Adjusted so the transition is smooth but decisive around the threshold.
    scale = sigma

    # Sigmoid function: 1 / (1 + exp( (x - threshold) / scale ))
    # If x is much lower than threshold (Attack), exponent is negative large, exp is small, Prob -> 1.
    # If x is much higher than threshold (Clean), exponent is positive large, exp is large, Prob -> 0.

    # We use (x - threshold) because low scores = attack.
    # Positive exponent makes denominator large -> Prob 0 (Correct for clean/high scores)

    # Calculate Z-score distance from the "safety threshold"
    z_dist = (scores - threshold) / (scale * 0.5)  # 0.5 factor to sharpen the transition slightly

    probabilities = 1 / (1 + np.exp(z_dist))

    return probabilities


def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    :param y_true: True binary labels (0: Clean, 1: Attacked)
    :param y_scores: Predicted probabilities of being attacked
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('./Graph_Tests/roc_curve.png')
    print("Saved roc_curve.png")
    plt.show()
    plt.close()