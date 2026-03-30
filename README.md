# Iterative Methods for Image Deblurring

This repository studies **iterative methods for image deblurring** on an ill-conditioned inverse problem.  
The project began with MATLAB implementations developed during a Directed Reading Program and was extended in Python to support a more complete data-based workflow, including:

- Image-quality metrics such as **PSNR** and **SSIM**
- Residual and error tracking across iterations
- Runtime and noise statistics
- CSV, Excel, and SQL analysis
- Plots for convergence comparison and restoration quality

The goal is not only to implement the algorithms, but also to analyze them the way an applied math or data-analysis project would be evaluated in practice: 
- **What is the starting point?**
- **What improves/degrades?**
- **What tradeoffs appear?**
- **Which method is better under which standard?**

---

## Project Motivation

As a classical **inverse problem**, image deblurring focuses on a blurred and noisy observation of the original image.
We hope to restore the clean image as accurately and precisely as possible

In this project, the deblurring model is studies through several iterative methods: Successive Over-relaxation (SOR); Steepst Descent Method; Conjugate Gradient Method.
Not suprisingly, since the restoration task is challenging due to its **ill-condition**, different iterative methods may behave differently in terms of:

- convergence behavior
- image quality
- numerical stability
- sensitivity to noise
- stopping behavior
- 
---

## Methods Involved

### 1. SOR Richardson Preconditioning
This serves as the **starting method** in this project.

It is relatively straightforward, but it uses a fixed relaxation parameter, or step size, and may require many iterations to reach expected image quality.

### 2. Steepest Descent Method
This method improves on SOR by using an **adaptive step size** at each iteration.

Compared to SOR, it overall reaches useful restoration quality in fewer iterations while remaining relatively stable.

### 3. Conjugate Gradient Method
This method is the most "aggressive" in terms of convergence speed.

It achieves strong early restoration quality very quickly. However, the results also show that it may require **early stopping** to avoid late-iteration deterioration in noisy inverse problems.

---

## Metrics Tracked

The Python implementations extend the original MATLAB work by tracking and exporting quantitative metrics throughout the iteration process.

### Common metrics
- **Residual norm**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **Runtime**
- **Noise mean / noise standard deviation**
- **Relative noise percentage**
- **Observation SNR**

### Method-specific metrics
- **Steepest Descent:** adaptive step size `tau`
- **Conjugate Gradient:** `beta`, `gamma`, relative reconstruction error

These metrics make it possible to compare methods not only by visual output, but also by measurable performance trends.

---

## Main Findings from the Current Experiment

The current outputs suggest the following interpretation:

- **SOR** provides a reasonable baseline, but it is comparatively slow.
- **Steepest Descent** is the most balanced improvement over the baseline:
  - it reaches strong PSNR much sooner than SOR
  - it keeps competitive image quality
  - it is easier to justify as a practical upgrade
- **Conjugate Gradient** improves fastest in the early stage, but it is also the most sensitive to over-iteration:
  - it reaches near-peak image quality in very few iterations
  - however, late iterations may degrade reconstruction quality, so stopping criteria matter

### Example takeaways from the current data
- Steepest Descent reaches **PSNR = 25** in about **half the iterations** needed by SOR.
- Conjugate Gradient reaches high-quality reconstructions the fastest, but the full iteration history shows that **smaller residuals do not always mean better perceptual image quality**.
- This project highlights an important lesson in inverse problems:  
  **the fastest numerical method is not automatically the best practical reconstruction method unless stopping rules are chosen carefully.**

---

## Repository Structure

```text
iterative-image-deblurring-methods/
│
├── Code/
│   ├── Cimmino_Diagonal_Algorithm.m
│   ├── SOR_Method.m
│   ├── Steepest_Descent_Method.m
│   ├── Conjugate_Gradient_Method.m
│   ├── SOR_metrics_runtime_noise.py
│   ├── Steepest_Descent_metrics_runtime_noise.py
│   ├── Conjugate_Gradient_metrics_runtime_noise.py
│   └── image_deblurring_data.mat
│
├── CSV-tables/
│   ├── SOR_Iteration_History.csv
│   ├── SOR_Metrics_Summary.csv
│   ├── Steepest_Descent_Iteration_History.csv
│   ├── Steepest_Descent_Metrics_Summary.csv
│   ├── Conjugate_Gradient_Iteration_History.csv
│   └── Conjugate_Gradient_Metrics_Summary.csv
│
├── SOR-plot/
├── Steepest-plot/
├── CG-plot/
│
├── sql-data/
│   ├── Iterative-Method-Analysis.sql
│   ├── Iterative-Method-Analysis.json
│   └── database.sqlite
│
├── Iterative_Algorithm_outputs.csv
└── Iterative_Algorithm_outputs.xlsx
