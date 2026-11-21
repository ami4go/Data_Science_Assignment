# Sketched Linear Regression via Johnson-Lindenstrauss (CountSketch)

An empirical evaluation of CountSketch dimensionality reduction for large-scale Ordinary Least Squares (OLS) regression on the **KDD Cup 1999** dataset ($n = 494{,}021$, $d = 116$, $\operatorname{rank}(D) = 113$).

## Overview

Solving overdetermined least squares problems $\min_x \|Dx - y\|_2^2$ directly scales as $O(nd^2)$. This project applies sparse CountSketch embeddings $S \in \mathbb{R}^{m \times n}$ to solve the sketched problem $\min_x \|S(Dx - y)\|_2^2$ in $O(\text{nnz}(D) + md^2)$ time and evaluates the approximation-runtime trade-off across sketch dimensions $m \in \{50, 100, 200, 500, 1000, 2000, 5000\}$ (5 random seeds per $m$).

## Key Findings

* **Exact Baseline:** Full QR/SVD least squares achieves $\text{Loss}(D, y, b) = 2949.7888$ in `3.662 s`.
* **Approximation Phase Transition:** For $m < \operatorname{rank}(D) = 113$, the sketched system is underdetermined and incurs large excess loss; once $m \ge 500$ ($\approx 4.4 \times \operatorname{rank}(D)$), the loss ratio stabilizes near $1.0$.
* **Speed-Up:** CountSketch achieves a $2.2\times\text{–}3.2\times$ wall-clock speed-up (`1.16 s – 1.69 s`), remaining nearly flat across $m$ because sparse hashing $O(\text{nnz}(D))$ dominates over solving the small $m \times d$ system.
* **Subspace Embedding:** Residual norm distortion $\left|\frac{\|S(Dx - y)\|_2}{\|Dx - y\|_2} - 1\right|$ decays at the theoretical $O(1/\sqrt{m})$ rate across candidate solutions.

## Requirements & Usage

* **Dependencies:** `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `jupyter`
* **Run:** Open the notebook and execute all cells (`kddcup99` is fetched automatically via `sklearn.datasets.fetch_kddcup99`).
