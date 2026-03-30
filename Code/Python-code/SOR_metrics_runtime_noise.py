"""
Successive Over-Relaxation

We adapted the SOR method from the MATLAB code to Python with the following iterative structure:

    M = D + tau * T,
    d = M \\ v,
    x_{k+1} = x_k + tau * d,

where `A^T A = T + D + T^T`.

The same structure is preserved. However, unlike MATLAB, we do not explicitly form A since it is a huge matrix and thus expensive.
Instead, since the blur is a 1D horizontal kernel, we shall work row by row and consider the convolution matrix H for one single row.

Added in this version:
- PSNR history
- SSIM history
- runtime tracking
- noise statistics
- final PSNR/SSIM
- plots for residual, PSNR, and SSIM
- CSV export

Files needed in the same folder to run this program:
- image_deblurring_data.mat

Run:
    python Successive_Over_Relaxation_with_metrics.py
"""

from __future__ import annotations

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import csv

from pathlib import Path

from scipy.io import loadmat

from scipy.linalg import convolution_matrix, solve_triangular

from skimage.metrics import structural_similarity as ssim

from time import perf_counter

# -----------------------------------------------------------------------------
# Helper function 1: Load the data from the MATLAB file and return what we need
# -----------------------------------------------------------------------------
def load_deblurring_data(mat_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the variables from the MATLAB data file.

    Returns:
    1) original_image : np.ndarray
        The original image, size (m, n).
    2) blurred_noisy_image : np.ndarray
        The noisy observation, size (m, n_full).
    3) psf : np.ndarray
        The point spread function stored as a 1D array.
    4) blurred_image_full : np.ndarray
        The noiseless blurred image for reference.
    """

    data = loadmat(mat_file)

    original_image = data["original_image"].astype(float)
    blurred_noisy_image = data["blurred_noisy_image"].astype(float)
    blurred_image_full = data["blurred_image_full"].astype(float)
    psf = data["PSF"].ravel().astype(float)

    return original_image, blurred_noisy_image, psf, blurred_image_full

# -----------------------------------------------------------------------------
# Helper function 2: Build the small matrices used by the iteration
# -----------------------------------------------------------------------------
def SORs(n: int, psf: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the matrices used in the Python SOR.

    Returns:
    1) A : np.ndarray
        1D full-convolution matrix, size (n_full, n).
    2) K : np.ndarray
        Row normal-equation matrix A^T * A, size (n, n).
    3) M : np.ndarray
        Lower triangular SOR matrix diag(diag(K)) + tau * tril(K, -1).
    """

    A = convolution_matrix(psf, n, mode="full")
    K = A.T @ A
    diag_K = np.diag(np.diag(K))
    lower_K = np.tril(K, k=-1)
    M = diag_K + tau * lower_K

    return A, K, M

# -----------------------------------------------------------------------------
# Helper function 3: Clip the reconstructed image to the original image range
# -----------------------------------------------------------------------------
def clip_to_reference_range(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Clip image values to the minimum/maximum values of the reference image.

    This makes PSNR and SSIM more meaningful when the iterate leaves the valid
    intensity range.
    """
    
    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))
    return np.clip(image, ref_min, ref_max)

# -----------------------------------------------------------------------------
# Helper function 4: Compute PSNR
# -----------------------------------------------------------------------------
def compute_psnr(reference: np.ndarray, image: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between a reference image and an
    estimated image.
    """

    clipped = clip_to_reference_range(reference, image)
    mse = float(np.mean((reference - clipped) ** 2))

    if mse == 0.0:
        return float("inf")

    data_range = float(np.max(reference) - np.min(reference))
    if data_range == 0.0:
        data_range = 1.0

    return 10.0 * np.log10((data_range ** 2) / mse)

# -----------------------------------------------------------------------------
# Helper function 5: Compute SSIM
# -----------------------------------------------------------------------------
def compute_ssim(reference: np.ndarray, image: np.ndarray) -> float:
    """
    Compute SSIM between the reference image and a reconstructed image.

    We clip the reconstructed image to the reference intensity range first,
    so the comparison stays meaningful even if an iterate is out of bounds.
    """
    ref = reference.astype(np.float64)
    img = clip_to_reference_range(reference, image).astype(np.float64)

    data_range = float(np.max(ref) - np.min(ref))
    if data_range == 0.0:
        data_range = 1.0

    score = ssim(
        ref,
        img,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return float(score)

# -----------------------------------------------------------------------------
# Helper function 6: Quantify the noise already present in the observed image
# -----------------------------------------------------------------------------
def compute_noise_statistics(clean_observation: np.ndarray, noisy_observation: np.ndarray) -> dict[str, np.ndarray | float]:
    """
    Compute exact noise statistics when both the clean blurred image and the
    noisy blurred image are available.

    The added noise is
        noise = noisy_observation - clean_observation.
    """

    noise = noisy_observation - clean_observation

    noise_mean = float(np.mean(noise))
    noise_std = float(np.std(noise))
    noise_fro_norm = float(np.linalg.norm(noise, ord="fro"))
    clean_fro_norm = float(np.linalg.norm(clean_observation, ord="fro"))

    if clean_fro_norm == 0.0:
        relative_noise_level = float("inf")
        observation_snr_db = float("inf")
    else:
        relative_noise_level = noise_fro_norm / clean_fro_norm
        if noise_fro_norm == 0.0:
            observation_snr_db = float("inf")
        else:
            observation_snr_db = 20.0 * np.log10(clean_fro_norm / noise_fro_norm)

    return {
        "noise_image": noise,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "noise_fro_norm": noise_fro_norm,
        "clean_blurred_fro_norm": clean_fro_norm,
        "relative_noise_level": relative_noise_level,
        "relative_noise_percent": 100.0 * relative_noise_level,
        "observation_snr_db": observation_snr_db,
    }

# -----------------------------------------------------------------------------
# Helper function 7: PNG-convert
# -----------------------------------------------------------------------------
def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a floating-point image to uint8 format by scaling to [0, 255].

    This is useful for saving or displaying images that are originally in a
    different range.
    """

    img = np.asarray(image, dtype=float)
    img_min = float(np.min(img))
    img_max = float(np.max(img))

    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)

    scaled = (img - img_min) / (img_max - img_min)
    uint8_image = (scaled * 255.0).round().astype(np.uint8)

    return uint8_image

# -----------------------------------------------------------------------------
# Helper function 7: Iteration data Export CSV
# -----------------------------------------------------------------------------
def export_metrices_to_csv(
        residual_history: list[float],
        psnr_history: list[float],
        ssim_history: list[float],
        max_iters: int,
        tol: float,
        runtime:float,
        filename: str = "SOR_Iteration_History.csv",
) -> None:
    """
    Export the residual, PSNR, SSIM history, etc. to a CSV file for later data analysis

    The CSV will have columns: Iteration, Residual, PSNR, SSIM.
    """

    if (not len(residual_history) == len(psnr_history) == len(ssim_history)):
        raise ValueError("All metric histories must have the same length.")

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Iteration", "Residual", "PSNR", "SSIM"])

        for i in range(len(residual_history)):
            writer.writerow([
                i + 1, 
                f"{residual_history[i]:.3f}",
                f"{psnr_history[i]:.3f}",
                f"{ssim_history[i]:.3f}"
            ])

    print(f"Metrics history exported to {filename}"
          f"(max_iters={max_iters}, tol={tol}, runtime={runtime:.4f} seconds)")
    
# -----------------------------------------------------------------------------
# Helper function 8: Extra data export to CSV
# -----------------------------------------------------------------------------
def export_extra_to_csv(
        tau: float,
        tol: float,
        final_psnr: float,
        final_ssim: float,
        noise_mean: float,
        noise_std: float,
        relative_noise_percent: float,
        observation_snr_db: float,
        best_psnr_iteration: int,
        best_ssim_iteration: int,
        filename: str = "SOR_Metrics_Summary.csv",
) -> None:
    """
    Export the final metrics summary to a CSV file for later data analysis

    The CSV will have columns: tau, tol, final_psnr, final_ssim, noise_mean, noise_std, relative_noise_percent, observation_snr_db, best_psnr_iteration, best_ssim_iteration.
    """

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow([
            "tau",
            "tol",
            "final_psnr",
            "final_ssim",
            "noise_mean",
            "noise_std",
            "relative_noise_percent",
            "observation_snr_db",
            "best_psnr_iteration",
            "best_ssim_iteration"
        ])

        writer.writerow([
            f"{tau:.3f}",
            f"{tol:.3f}",
            f"{final_psnr:.3f}",
            f"{final_ssim:.3f}",
            f"{noise_mean:.6f}",
            f"{noise_std:.6f}",
            f"{relative_noise_percent:.3f}",
            f"{observation_snr_db:.3f}",
            best_psnr_iteration,
            best_ssim_iteration
        ])

    print(f"Metrics summary exported to {filename}"
          f"(tau={tau}, tol={tol}, final_psnr={final_psnr:.4f}, final_ssim={final_ssim:.4f})")
# -----------------------------------------------------------------------------
# Main deblurring process using SOR iteration
# -----------------------------------------------------------------------------
def SOR_deblur(
    mat_file: str = "image_deblurring_data.mat",
    tau: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-2,
    print_every: int = 10,
) -> tuple[np.ndarray, list[float], dict[str, np.ndarray | list[float] | float]]:
    """
    Deblur the image using the same SOR iteration pattern as the MATLAB code.

    Returns:
    1) x0 : np.ndarray
        Recovered image of size (m, n).
    2) residual_history : list[float]
        Track the residual at each iteration.
    3) extras : dict
        Useful arrays and metric histories for later plotting and analysis.
    """

    total_start = perf_counter()

    original_image, blurred_noisy_image, psf, blurred_image_full = load_deblurring_data(mat_file)

    m, n = original_image.shape
    m_full, _ = blurred_noisy_image.shape

    if m_full != m:
        raise ValueError(
            "This script assumes the attached blur acts row-by-row. "
            f"Got original shape {(m, n)} and blurred shape {blurred_noisy_image.shape}."
        )

    A, K, M = SORs(n=n, psf=psf, tau=tau)
    noise_stats = compute_noise_statistics(blurred_image_full, blurred_noisy_image)

    # Initial guess x0 = 0, stored as an image instead of a long vector
    x0 = np.zeros((m, n), dtype=float)

    residual_history: list[float] = []
    psnr_history: list[float] = []
    ssim_history: list[float] = []

    iteration_start = perf_counter()

    for k in range(max_iters):
        # Blur of the current iterate x_k
        Ax0 = x0 @ A.T

        # Residual at iteration k: r_k = b - A * x_k
        r = blurred_noisy_image - Ax0
        residual_norm = float(np.linalg.norm(r, ord="fro"))
        residual_history.append(residual_norm)

        # Quality of the current iterate compared with the original image
        current_psnr = compute_psnr(original_image, x0)
        current_ssim = compute_ssim(original_image, x0)
        psnr_history.append(current_psnr)
        ssim_history.append(current_ssim)

        if (k + 1) % print_every == 0 or k == 0:
            print(
                f"Iteration {k+1:4d} | residual = {residual_norm:.6f} "
                f"| PSNR = {current_psnr:.4f} | SSIM = {current_ssim:.4f}"
            )

        if residual_norm <= tol:
            print(f"Converged at iteration {k+1} with residual {residual_norm:.6f}")
            break

        # Compute v_k = A^T r_k in row form
        V = r @ A

        # Solve M d_k^T = v_k^T for all rows at once
        D = solve_triangular(M, V.T, lower=True).T

        # Update x_{k+1} = x_k + tau * d_k
        x0 = x0 + tau * D

    iteration_runtime_seconds = perf_counter() - iteration_start
    total_runtime_seconds = perf_counter() - total_start

    final_psnr = compute_psnr(original_image, x0)
    final_ssim = compute_ssim(original_image, x0)

    # Best iterations with respect to each quality metric
    best_psnr_idx = int(np.argmax(psnr_history)) + 1
    best_ssim_idx = int(np.argmax(ssim_history)) + 1

    extras = {
        "original_image": original_image,
        "blurred_noisy_image": blurred_noisy_image,
        "blurred_image_full": blurred_image_full,
        "A": A,
        "K": K,
        "M": M,
        "psnr_history": psnr_history,
        "ssim_history": ssim_history,
        "final_psnr": final_psnr,
        "final_ssim": final_ssim,
        "best_psnr_iteration": best_psnr_idx,
        "best_ssim_iteration": best_ssim_idx,
        "iteration_runtime_seconds": iteration_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "average_time_per_iteration": iteration_runtime_seconds / max(len(residual_history), 1),
        **noise_stats,
    }

    return x0, residual_history, extras

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(X: np.ndarray, residual_history: list[float], extras: dict[str, np.ndarray | list[float] | float]) -> None:
    """
    Display the observed image, recovered image, original image,
    and the residual/PSNR/SSIM histories.
    """

    observed = extras["blurred_noisy_image"]
    original = extras["original_image"]
    psnr_history = extras["psnr_history"]
    ssim_history = extras["ssim_history"]
    final_psnr = extras["final_psnr"]
    final_ssim = extras["final_ssim"]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(observed, cmap="gray")
    plt.title("Blurred Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(X, cmap="gray")
    plt.title(f"Recovered Image\nPSNR = {final_psnr:.2f}, SSIM = {final_ssim:.4f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(residual_history)
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm")
    plt.title("SOR Residual History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(psnr_history)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("SOR PSNR History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(ssim_history)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("SOR SSIM History")
    plt.tight_layout()

    plt.show()

def main() -> None:
    """
    Run the full deblurring program on the MATLAB data.
    """

    tau = 0.1
    max_iters = 200
    tol = 1e-2

    x0, residual_history, extras = SOR_deblur(
        mat_file="image_deblurring_data.mat",
        tau=tau,
        max_iters=max_iters,
        tol=tol,
        print_every=10,
    )

    export_metrices_to_csv(
        residual_history=residual_history,
        psnr_history=extras["psnr_history"],
        ssim_history=extras["ssim_history"],
        max_iters=max_iters,
        tol=tol,
        runtime=extras["total_runtime_seconds"],
    )
    export_extra_to_csv(
        tau=tau,
        tol=tol,
        final_psnr=extras["final_psnr"],
        final_ssim=extras["final_ssim"],
        noise_mean=extras["noise_mean"],
        noise_std=extras["noise_std"],
        relative_noise_percent=extras["relative_noise_percent"],
        observation_snr_db=extras["observation_snr_db"],
        best_psnr_iteration=extras["best_psnr_iteration"],
        best_ssim_iteration=extras["best_ssim_iteration"],
    )

    print("\nFinished.")
    print(f"Number of iterations performed: {len(residual_history)}")
    print(f"Initial residual: {residual_history[0]:.6f}")
    print(f"Final residual:   {residual_history[-1]:.6f}")
    print(f"Final PSNR:       {extras['final_psnr']:.4f} dB")
    print(f"Final SSIM:       {extras['final_ssim']:.4f}")
    print(f"Total runtime:    {extras['total_runtime_seconds']:.4f} seconds")
    print(f"Iter. runtime:    {extras['iteration_runtime_seconds']:.4f} seconds")
    print(f"Avg/iteration:    {extras['average_time_per_iteration']:.6f} seconds")
    print(f"Noise mean:       {extras['noise_mean']:.6f}")
    print(f"Noise std:        {extras['noise_std']:.6f}")
   # print(f"Noise Fro norm:   {extras['noise_fro_norm']:.6f}")
    print(f"Relative noise:   {extras['relative_noise_percent']:.4f}%")
    print(f"Observation SNR:  {extras['observation_snr_db']:.4f} dB")
    print(f"Best PSNR iteration in history: {extras['best_psnr_iteration']}")
    print(f"Best SSIM iteration in history: {extras['best_ssim_iteration']}")

    plot_results(x0, residual_history, extras)

if __name__ == "__main__":
    main()