"""
Conjugate Gradient Method

We adapt the MATLAB Conjugate Gradient method to Python based on Golb-Kahan Bidiagonalization (GKB) with the following row-wiseiterative structure:

    beta_1 = ||g||,
    w_1 = g / beta_1,
    y_hat_1 = A^T w_1,
    gamma_1 = ||y_hat_1||,
    y_1 = y_hat_1 / gamma_1,

and for k = 2, 3, ...,

    w_hat_k = A y_{k-1} - gamma_{k-1} w_{k-1},
    beta_k = ||w_hat_k||,  
    w_k = w_hat_k / beta_k,
    y_hat_k = A^T w_k - beta_k y_{k-1},
    gamma_k = ||y_hat_k||,
    y_k = y_hat_k / gamma_k.

After the GBC process, we have the bidiagonal matrix B_k, solve the adapated least squares problem to get f_hat,
and thus reconstruct the image 
    f(:,p) = sum_{j=1}^p f_hat(j) * y_j for p = 1, 2, ..., k_final.

Metrics and outputs tracked in this algorithm:
- residue history
- image error history
- PSNR history
- SSIM history
- beta and gamma history
- noise statistics
- runtime
- final PSNR/SSIM
- plots for residual, image error, PSNR, and SSIM
- CSV export

Files needed in the same folder to run this program:
- image_deblurring_data.mat

Run:
    python Conjugate_Gradient_metrics_runtime_noise.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from time import perf_counter

from scipy.io import loadmat

from scipy.linalg import convolution_matrix

from skimage.metrics import structural_similarity as ssim

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
# Helper function 2: Build the small matrix used by the row-wise iteration
# -----------------------------------------------------------------------------
def CGs(n: int, psf: np.ndarray) -> np.ndarray:
    """
    Build the small matrix used in the Python conjugate gradient method.

    Returns:
    1) K : np.ndarray
        1D full-convolution matrix, size (n_full, n).
    """

    K = convolution_matrix(psf, n, mode="full")

    return K

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
# Helper function 8: Check whether the CSV file is empty before writing header
# -----------------------------------------------------------------------------
def should_write_header(filename: str) -> bool:
    """
    Return True if the CSV file does not exist yet or is empty.
    """

    file_path = Path(filename)
    return (not file_path.exists()) or file_path.stat().st_size == 0

# -----------------------------------------------------------------------------
# Helper function 9: Iteration data Export CSV
# -----------------------------------------------------------------------------
def export_metrices_to_csv(
    beta_history: list[float],
    gamma_history: list[float],
    residual_history: list[float],
    relative_error_history: list[float],
    psnr_history: list[float],
    ssim_history: list[float],
    runtime: float,
    filename: str = "Conjugate_Gradient_Iteration_History.csv",
) -> None:
    """
    Export the beta, gamma, residual, relative error, PSNR, and SSIM history
    to a CSV file for later data analysis.

    The CSV will have columns:
    Iteration, Beta, Gamma, Residual, Relative_Error, PSNR, SSIM.
    """

    if not (
        len(beta_history)
        == len(gamma_history)
        == len(residual_history)
        == len(relative_error_history)
        == len(psnr_history)
        == len(ssim_history)
    ):
        raise ValueError("All metric histories must have the same length.")

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Iteration",
            "Beta",
            "Gamma",
            "Residual",
            "Relative_Error",
            "PSNR",
            "SSIM",
        ])

        for i in range(len(residual_history)):
            writer.writerow([
                i + 1,
                f"{beta_history[i]:.3f}",
                f"{gamma_history[i]:.3f}",
                f"{residual_history[i]:.3f}",
                f"{relative_error_history[i]:.6f}",
                f"{psnr_history[i]:.3f}",
                f"{ssim_history[i]:.3f}",
            ])

    print(
        f"Metrics history exported to {filename} "
        f"(runtime={runtime:.4f} seconds)"
    )

# -----------------------------------------------------------------------------
# Helper function 10: Extra data export to CSV
# -----------------------------------------------------------------------------
def export_extra_to_csv(
    max_iters: int,
    k_final: int,
    best_p: int,
    final_residual: float,
    final_relative_error: float,
    final_psnr: float,
    final_ssim: float,
    noise_mean: float,
    noise_std: float,
    relative_noise_percent: float,
    observation_snr_db: float,
    best_psnr_iteration: int,
    best_ssim_iteration: int,
    filename: str = "Conjugate_Gradient_Metrics_Summary.csv",
) -> None:
    """
    Export the final metrics summary to a CSV file for later data analysis.

    The CSV will have columns:
    max_iters, k_final, best_p, final_residual, final_relative_error,
    final_psnr, final_ssim, noise_mean, noise_std,
    relative_noise_percent, observation_snr_db,
    best_psnr_iteration, best_ssim_iteration.
    """

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "max_iters",
            "k_final",
            "best_p",
            "final_residual",
            "final_relative_error",
            "final_psnr",
            "final_ssim",
            "noise_mean",
            "noise_std",
            "relative_noise_percent",
            "observation_snr_db",
            "best_psnr_iteration",
            "best_ssim_iteration",
        ])

        writer.writerow([
            max_iters,
            k_final,
            best_p,
            f"{final_residual:.3f}",
            f"{final_relative_error:.6f}",
            f"{final_psnr:.3f}",
            f"{final_ssim:.3f}",
            f"{noise_mean:.6f}",
            f"{noise_std:.6f}",
            f"{relative_noise_percent:.3f}",
            f"{observation_snr_db:.3f}",
            best_psnr_iteration,
            best_ssim_iteration,
        ])

    print(
        f"Metrics summary exported to {filename} "
        f"(final_psnr={final_psnr:.4f}, final_ssim={final_ssim:.4f})"
    )

# -----------------------------------------------------------------------------
# Main deblurring process using conjugate gradient method
# -----------------------------------------------------------------------------
def conjugate_gradient_deblur(
    mat_file: str = "image_deblurring_data.mat",
    max_iters: int = 200,
    print_every: int = 10,
) -> tuple[np.ndarray, list[float], dict[str, np.ndarray | list[float] | float]]:
    """
    Deblur the image using the same conjugate gradient pattern as the MATLAB code.

    Returns:
    1) recovered_image : np.ndarray
        Recovered image of size (m, n).
    2) residual_history : list[float]
        Track the data residual norm for each reconstructed image f(:,p).
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

    K = CGs(n=n, psf=psf)
    noise_stats = compute_noise_statistics(blurred_image_full, blurred_noisy_image)

    max_iters = min(m, n, max_iters)

    g = blurred_noisy_image.astype(float)

    beta = np.zeros(max_iters + 2, dtype=float)
    gamma = np.zeros(max_iters + 2, dtype=float)

    w: list[np.ndarray | None] = [None] * (max_iters + 2)
    w_hat: list[np.ndarray | None] = [None] * (max_iters + 2)
    y: list[np.ndarray | None] = [None] * (max_iters + 2)
    y_hat: list[np.ndarray | None] = [None] * (max_iters + 2)

    beta[1] = float(np.linalg.norm(g, ord="fro"))
    if beta[1] == 0.0:
        raise ValueError("The observed image is zero, so the initialization beta(1) is zero.")

    w[1] = g / beta[1]
    y_hat[1] = w[1] @ K
    gamma[1] = float(np.linalg.norm(y_hat[1], ord="fro"))
    if gamma[1] == 0.0:
        raise ValueError("gamma(1) is zero, so the first basis image cannot be normalized.")

    y[1] = y_hat[1] / gamma[1]

    k_final = 1

    iteration_start = perf_counter()

    for k in range(2, max_iters + 1):
        w_hat[k] = y[k - 1] @ K.T - gamma[k - 1] * w[k - 1]
        beta[k] = float(np.linalg.norm(w_hat[k], ord="fro"))

        if beta[k] == 0.0:
            k_final = k - 1
            print(f"Stopped GKB at iteration {k}: beta became zero.")
            break

        w[k] = w_hat[k] / beta[k]

        y_hat[k] = w[k] @ K - beta[k] * y[k - 1]
        gamma[k] = float(np.linalg.norm(y_hat[k], ord="fro"))

        if gamma[k] == 0.0:
            y[k] = np.zeros_like(y_hat[k])
            k_final = k
            print(f"Stopped GKB at iteration {k}: gamma became zero.")
            break

        y[k] = y_hat[k] / gamma[k]
        k_final = k

        if k % print_every == 0 or k == 2:
            print(
                f"GKB iteration {k:4d} | beta = {beta[k]:.6f} | gamma = {gamma[k]:.6f}"
            )

    residual_history: list[float] = []
    relative_error_history: list[float] = []
    psnr_history: list[float] = []
    ssim_history: list[float] = []
    beta_history: list[float] = []
    gamma_history: list[float] = []

    f: list[np.ndarray | None] = [None] * (k_final + 1)

    err = float("inf")
    best_p = 1

    g_norm = float(np.linalg.norm(g, ord="fro"))
    original_norm = float(np.linalg.norm(original_image, ord="fro"))

    for p in range(1, k_final + 1):
        B_k = np.zeros((p + 1, p), dtype=float)

        for i in range(1, p + 1):
            B_k[i - 1, i - 1] = gamma[i]

        for j in range(2, p + 2):
            B_k[j - 1, j - 2] = beta[j]

        suv = np.zeros(p + 1, dtype=float)
        suv[0] = g_norm

        f_hat = np.linalg.lstsq(B_k, suv, rcond=None)[0]

        f[p] = np.zeros((m, n), dtype=float)
        for j in range(1, p + 1):
            f[p] = f[p] + f_hat[j - 1] * y[j]

        err_update = float(np.linalg.norm(f[p] - original_image, ord="fro") / original_norm)
        residual_update = float(np.linalg.norm(g - (f[p] @ K.T), ord="fro"))
        current_psnr = compute_psnr(original_image, f[p])
        current_ssim = compute_ssim(original_image, f[p])

        residual_history.append(residual_update)
        relative_error_history.append(err_update)
        psnr_history.append(current_psnr)
        ssim_history.append(current_ssim)
        beta_history.append(float(beta[p]))
        gamma_history.append(float(gamma[p]))

        if err_update < err:
            err = err_update
            best_p = p

        if p % print_every == 0 or p == 1:
            print(
                f"Reconstruction {p:4d} | residual = {residual_update:.6f} "
                f"| rel. error = {err_update:.6f} | PSNR = {current_psnr:.4f} "
                f"| SSIM = {current_ssim:.4f}"
            )

    iteration_runtime_seconds = perf_counter() - iteration_start
    total_runtime_seconds = perf_counter() - total_start

    recovered_image = f[best_p]
    final_residual = residual_history[best_p - 1]
    final_relative_error = relative_error_history[best_p - 1]
    final_psnr = psnr_history[best_p - 1]
    final_ssim = ssim_history[best_p - 1]

    best_psnr_iteration = int(np.argmax(psnr_history)) + 1
    best_ssim_iteration = int(np.argmax(ssim_history)) + 1

    extras = {
        "original_image": original_image,
        "blurred_noisy_image": blurred_noisy_image,
        "blurred_image_full": blurred_image_full,
        "K": K,
        "beta_history": beta_history,
        "gamma_history": gamma_history,
        "relative_error_history": relative_error_history,
        "psnr_history": psnr_history,
        "ssim_history": ssim_history,
        "final_residual": final_residual,
        "final_relative_error": final_relative_error,
        "final_psnr": final_psnr,
        "final_ssim": final_ssim,
        "best_p": best_p,
        "k_final": k_final,
        "best_psnr_iteration": best_psnr_iteration,
        "best_ssim_iteration": best_ssim_iteration,
        "iteration_runtime_seconds": iteration_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "average_time_per_iteration": iteration_runtime_seconds / max(k_final, 1),
        **noise_stats,
    }

    return recovered_image, residual_history, extras

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(X: np.ndarray, residual_history: list[float], extras: dict[str, np.ndarray | list[float] | float]) -> None:
    """
    Display the observed image, recovered image, original image,
    and the residual/beta/gamma/PSNR/SSIM histories.
    """

    observed = extras["blurred_noisy_image"]
    original = extras["original_image"]
    beta_history = extras["beta_history"]
    gamma_history = extras["gamma_history"]
    relative_error_history = extras["relative_error_history"]
    psnr_history = extras["psnr_history"]
    ssim_history = extras["ssim_history"]
    final_psnr = extras["final_psnr"]
    final_ssim = extras["final_ssim"]
    best_p = extras["best_p"]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(observed, cmap="gray")
    plt.title("Blurred Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(X, cmap="gray")
    plt.title(f"Recovered Image\nBest p = {best_p}, PSNR = {final_psnr:.2f}, SSIM = {final_ssim:.4f}")
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
    plt.title("Conjugate Gradient Residual History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(relative_error_history)
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title("Conjugate Gradient Relative Error History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(beta_history)
    plt.xlabel("Iteration")
    plt.ylabel("Beta")
    plt.title("Conjugate Gradient Beta History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(gamma_history)
    plt.xlabel("Iteration")
    plt.ylabel("Gamma")
    plt.title("Conjugate Gradient Gamma History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(psnr_history)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("Conjugate Gradient PSNR History")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.plot(ssim_history)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("Conjugate Gradient SSIM History")
    plt.tight_layout()

    plt.show()


def main() -> None:
    """
    Run the full deblurring program on the MATLAB data.
    """

    max_iters = 200

    recovered_image, residual_history, extras = conjugate_gradient_deblur(
        mat_file="image_deblurring_data.mat",
        max_iters=max_iters,
        print_every=10,
    )

    export_metrices_to_csv(
        beta_history=extras["beta_history"],
        gamma_history=extras["gamma_history"],
        residual_history=residual_history,
        relative_error_history=extras["relative_error_history"],
        psnr_history=extras["psnr_history"],
        ssim_history=extras["ssim_history"],
        runtime=extras["total_runtime_seconds"],
    )
    export_extra_to_csv(
        max_iters=max_iters,
        k_final=extras["k_final"],
        best_p=extras["best_p"],
        final_residual=extras["final_residual"],
        final_relative_error=extras["final_relative_error"],
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
    print(f"Number of reconstructed candidates: {len(residual_history)}")
    print(f"Best p:            {extras['best_p']}")
    print(f"Final residual:    {extras['final_residual']:.6f}")
    print(f"Final rel. error:  {extras['final_relative_error']:.6f}")
    print(f"Final PSNR:        {extras['final_psnr']:.4f} dB")
    print(f"Final SSIM:        {extras['final_ssim']:.4f}")
    print(f"Total runtime:     {extras['total_runtime_seconds']:.4f} seconds")
    print(f"Iter. runtime:     {extras['iteration_runtime_seconds']:.4f} seconds")
    print(f"Avg/iteration:     {extras['average_time_per_iteration']:.6f} seconds")
    print(f"Noise mean:        {extras['noise_mean']:.6f}")
    print(f"Noise std:         {extras['noise_std']:.6f}")
    print(f"Noise Fro norm:    {extras['noise_fro_norm']:.6f}")
    print(f"Relative noise:    {extras['relative_noise_percent']:.4f}%")
    print(f"Observation SNR:   {extras['observation_snr_db']:.4f} dB")
    print(f"Best PSNR iteration in history: {extras['best_psnr_iteration']}")
    print(f"Best SSIM iteration in history: {extras['best_ssim_iteration']}")

    plot_results(recovered_image, residual_history, extras)

if __name__ == "__main__":
    main()