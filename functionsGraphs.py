import numpy as np
import math
import os
import torch
import matplotlib.pyplot as plt
from functionsAttackDetection import fit_clean_distribution, calculate_attack_probability

# Avoid OpenMP issues
os.environ['OMP_NUM_THREADS'] = '1'


def plot_histograms(data, all_labels, x_label_str, y_label_str, filename, color_c='teal', color_a='orange'):
    """
    Generates figures for the requested histograms with optimized limits.
    Handles both Link-Level (required) and User-Level (optional) data.
    Standard histograms with separated labels: Clean on top, Attacked below X axis.
    """
    plt.figure(figsize=(10, 6))

    labels = np.asarray(all_labels)

    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')

    clean_idx = (labels == 0)
    attacked_idx = (labels == 1)

    # 1. Determine Dynamic Limits (Discretization & Clipping)
    if len(data) > 0:
        limit_upper = np.percentile(data, 98)
        limit_lower = np.min(data)
        # Create 50 evenly spaced bins
        bins = np.linspace(limit_lower, limit_upper, 100)
    else:
        bins = 50
        limit_lower, limit_upper = 0, 1

    # Dictionary to track max height to adjust ylim later
    max_height = 0

    # --- Plot histograms (Standard: Both upwards) ---
    kwargs = dict(alpha=0.6, bins=bins, density=True, histtype='stepfilled')

    # CLEAN HISTOGRAM
    counts_c, edges_c, _ = plt.hist(data[clean_idx], color=color_c, label="Clean pilot transmissions", **kwargs)
    max_height = max(max_height, max(counts_c))

    # # Annotate Clean Counts (ABOVE bar)
    # bin_width = edges_c[1] - edges_c[0]
    # for i in range(len(counts_c)):
    #     if counts_c[i] > 0:
    #         plt.text(edges_c[i] + bin_width / 2, counts_c[i], int(counts_c[i]),
    #                  ha='center', va='bottom', fontsize=8, color=color_c, fontweight='bold')

    # ATTACKED HISTOGRAM
    counts_a, edges_a, _ = plt.hist(data[attacked_idx], color=color_a, label="Attacked pilot transmissions", **kwargs)
    max_height = max(max_height, max(counts_a))

    # # Annotate Attacked Counts (BELOW X-axis, y=0)
    # bin_width = edges_a[1] - edges_a[0]
    # for i in range(len(counts_a)):
    #     if counts_a[i] > 0:
    #         # Place text just below 0 line
    #         plt.text(edges_a[i] + bin_width / 2, 0, int(counts_a[i]),
    #                  ha='center', va='top', fontsize=8, color=color_a, fontweight='bold')

    # # Draw a horizontal line at y=0
    # plt.axhline(0, color='black', linewidth=0.8)

    # Limit X-axis
    plt.xlim(limit_lower, limit_upper)

    # Expand Y-axis to accommodate labels at the bottom (negative space) and top
    plt.ylim(bottom=-max_height * 0.08, top=max_height * 1.1)

    plt.xlabel(x_label_str)
    plt.ylabel(y_label_str)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Saved {filename}")

    plt.show()
    plt.close()




def plot_scatter(x_values, y_values, all_labels, x_label_str, y_label_str, filename,
                 figsize=(10, 6), attacked_color='orange', clean_color='teal',
                 attacked_label='Attacked pilot transmissions', clean_label='Clean pilot transmissions', s=10, alpha=0.5):
    """
    Plotea un scatter 2D separado por etiquetas, acepta etiquetas en formato LaTeX
    y fuerza la fuente Times New Roman.

    Args:
        x_values, y_values, all_labels: array-like.
        x_label_str, y_label_str
        filename: ruta donde guardar la figura.
        use_usetex (bool): si True intenta usar `text.usetex` (requiere LaTeX instalado).
        font_family (str): nombre de la fuente serif a usar (por defecto Times New Roman).
        math_fontset (str): conjunto para mathtext si `usetex` no se usa (ej. 'stix').
    """
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    labels = np.asarray(all_labels)

    if x.shape[0] != y.shape[0] or x.shape[0] != labels.shape[0]:
        raise ValueError("`x_values`, `y_values` y `all_labels` deben tener la misma longitud.")

    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')

    clean_idx = (labels == 0)
    attacked_idx = (labels == 1)

    plt.figure(figsize=figsize)

    if clean_idx.any():
        plt.scatter(x[clean_idx], y[clean_idx],
                    color=clean_color, alpha=alpha, label=clean_label, s=s)

    if attacked_idx.any():
        plt.scatter(x[attacked_idx], y[attacked_idx],
                    color=attacked_color, alpha=alpha, label=attacked_label, s=s)

    plt.xlabel(x_label_str, size=15)
    plt.ylabel(y_label_str, size=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"Saved `{filename}`")
    plt.show()
    plt.close()


def plot_attack_probability(all_avg_kl, all_pilot_labels,
                               save_path='./Graphs/hist_attack_probability.png',
                               bins=100, show=True):
    """
    Fit a Gaussian to KL scores from clean pilots, compute attack probabilities,
    plot and save a histogram, and plot the ROC curve.

    Parameters
    - all_avg_kl: array-like, average KL scores per pilot
    - all_pilot_labels: array-like, labels per pilot (0 clean, 1 attacked)
    - save_path: path where the histogram will be saved
    - bins: number of bins for the histogram
    - show: if True, call plt.show() before closing the figure

    Returns
    - (probs, mu_clean, sigma_clean) on success
    - (None, None, None) if no clean samples are available to fit the detector
    """

    # Ensure numpy arrays
    all_avg_kl = np.asarray(all_avg_kl)
    all_pilot_labels = np.asarray(all_pilot_labels)

    # Boolean masks for clean and attacked pilots
    clean_idx = (all_pilot_labels == 0)
    attacked_idx = (all_pilot_labels == 1)

    # Extract KL scores for known-clean pilots
    clean_kl_scores = all_avg_kl[clean_idx]

    if len(clean_kl_scores) == 0:
        print("Error: No clean pilot samples available to fit the detector.")
        return None, None, None

    # Fit the clean distribution (external function)
    mu_clean, sigma_clean = fit_clean_distribution(clean_kl_scores)
    print(f"Clean distribution fitted: mean={mu_clean:.4f}, std={sigma_clean:.4f}")

    # Compute attack probabilities for all pilots (external function)
    probs = calculate_attack_probability(all_avg_kl, mu_clean, sigma_clean)

    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')

    # Plot histogram of probabilities for clean vs attacked pilots
    plt.figure(figsize=(10, 6))
    plt.hist(probs[clean_idx], color='teal', alpha=0.6, label='Clean pilot transmissions', bins=bins, density=True)
    plt.hist(probs[attacked_idx], color='orange', alpha=0.6, label='Attacked pilot transmissions', bins=bins, density=True)
    plt.xlabel(r"$P($Attack$ | s_{t_k}, \eta)$", size=15)
    plt.ylabel('Density', size=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved {save_path}")

    if show:
        plt.show()
    plt.close()

    return probs, mu_clean, sigma_clean


def plot_roc_curve(y_true, y_scores):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    :param y_true: True binary labels (0: Clean, 1: Attacked)
    :param y_scores: Predicted probabilities of being attacked
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=15)
    plt.ylabel('True Positive Rate', size=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('./Graphs/roc_curve.png', dpi=600, bbox_inches='tight')
    print("Saved roc_curve.png")
    plt.show()
    plt.close()


def plot_shapedKL_histogram(data, all_labels, x_label_str, y_label_str, filename, color_c='teal', color_a='orange'):
    """
    Generate histograms for labeled data and overlay the Gaussian PDF
    fitted to the clean samples. Adds a vertical dashed line at
    x = mu_clean - 2.5 * sigma_clean from y=0 to the gaussian curve.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from functionsAttackDetection import fit_clean_distribution
    import os

    plt.figure(figsize=(10, 6))

    labels = np.asarray(all_labels)
    values = np.asarray(data)

    # Use LaTeX text and Times New Roman font
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')

    clean_idx = (labels == 0)
    attacked_idx = (labels == 1)

    # Determine plotting limits and bins
    if values.size > 0:
        limit_upper = np.percentile(values, 98)
        limit_lower = np.min(values)
        bins = np.linspace(limit_lower, limit_upper, 100)
    else:
        bins = 50
        limit_lower, limit_upper = 0, 1

    max_height = 0

    # Histogram kwargs: density=True so PDF and hist align
    kwargs = dict(alpha=0.6, bins=bins, density=True, histtype='stepfilled')

    # Plot clean histogram (if any)
    if clean_idx.any():
        counts_c, edges_c, _ = plt.hist(values[clean_idx], color=color_c,
                                        label="Clean pilot transmissions", **kwargs)
        max_height = max(max_height, counts_c.max())
    else:
        counts_c = np.array([0.0])
        edges_c = np.linspace(limit_lower, limit_upper, 2)

    # Plot attacked histogram (if any)
    if attacked_idx.any():
        counts_a, edges_a, _ = plt.hist(values[attacked_idx], color=color_a,
                                        label="Attacked pilot transmissions", **kwargs)
        max_height = max(max_height, counts_a.max())
    else:
        counts_a = np.array([0.0])
        edges_a = np.linspace(limit_lower, limit_upper, 2)

    # Fit Gaussian to clean samples and overlay PDF
    if clean_idx.any():
        clean_values = values[clean_idx]
        mu_clean, sigma_clean = fit_clean_distribution(clean_values)

        # Only plot if sigma is positive and finite
        if np.isfinite(sigma_clean) and sigma_clean > 0:
            x_pdf = np.linspace(limit_lower, limit_upper, 1000)
            coef = 1.0 / (sigma_clean * np.sqrt(2.0 * np.pi))
            pdf = coef * np.exp(-0.5 * ((x_pdf - mu_clean) / sigma_clean) ** 2)

            # Update max height to include PDF peak
            max_height = max(max_height, float(np.max(pdf)))

            plt.plot(x_pdf, pdf, color='black', lw=2,
                     label=r'$\mathcal{N}(\mu_{\textsubscript{clean}}, \sigma_{\textsubscript{clean}}^2)$')

            # plt.plot(x_pdf, pdf, color='black', lw=2,
            #          label=r'Validation set s_{t_k} score distribution ($\mu_{\textsubscript{clean}}=$' + f'{mu_clean:.3f}' + r", $\sigma_{\textsubscript{clean}}=$" + f'{sigma_clean:.3f})')

            # Draw vertical line at mu_clean - 2.5*sigma_clean up to the gaussian curve
            x_marker = mu_clean - 2 * sigma_clean
            # compute pdf value at x_marker
            y_marker = coef * np.exp(-0.5 * ((x_marker - mu_clean) / sigma_clean) ** 2)

            # If marker is outside current x-limits, expand limits to include it
            if x_marker < limit_lower:
                limit_lower = x_marker - 0.02 * (limit_upper - x_marker)
            if x_marker > limit_upper:
                limit_upper = x_marker + 0.02 * (x_marker - limit_lower)

            # Ensure the y-axis accommodates the marker
            max_height = max(max_height, float(y_marker))

            # Plot dashed vertical line from y=0 to y=y_marker
            plt.plot([x_marker, x_marker], [0.0, y_marker], color='black', linestyle='--', linewidth=2)

            # Put a marker at the top and a small text label
            plt.scatter([x_marker], [y_marker], color='black', zorder=5, label=r"$\eta = \mu_{\textsubscript{clean}} - 2\sigma_{\textsubscript{clean}}$")
            # plt.annotate(f'{x_marker:.3f}', xy=(x_marker, y_marker), xytext=(5, 5),
            #              textcoords='offset points', fontsize=9, color='black')
        else:
            print("Warning: sigma for clean fit is non-positive or not finite; skipping Gaussian plot and marker.")
    else:
        print("Warning: no clean samples available; skipping Gaussian fit and marker.")

    # Limits and labels
    plt.xlim(limit_lower, limit_upper)
    plt.ylim(bottom=-max_height * 0.08, top=max_height * 1.1)

    plt.xlabel(x_label_str, size=15)
    plt.ylabel(y_label_str, size=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"Saved {filename}")

    plt.show()
    plt.close()

