# Inexact Bregman Sparse Newton Method for Efficient Optimal Transport

This repository contains the implementation of the Inexact Bregman Sparse Newton (IBSN) method for optimal transport (OT), together with baseline solvers and scripts for numerical experiments. The experiments cover synthetic and real datasets, entropically regularized optimal transport (EOT), and image color transfer.

## Repository Structure

| Path | Description |
| --- | --- |
| `Alg/` | IBSN, EOT solvers, and comparison algorithms. |
| `Alg/IBSN_Solver.py` | Main IBSN implementation. |
| `Alg/Algorithm_Import.py` | Shared algorithm imports, experiment wrappers, and result utilities. |
| `data/` | Benchmark datasets and image assets. |
| `Data_Sample_Import.py` | Dataset loading and OT problem construction. |
| `meta_ot/` | Supporting code for population-density experiments. |
| `Color_transfer/` | Image color-transfer experiment scripts. |
| `Results/` | Saved experiment results and figures. |
| `demo.py` | Standalone synthetic example using IBSN. |

## Quick Start

From the repository root, run:

```bash
python demo.py
```

The demo generates a random cost matrix of shape `500 × 500` and two normalized random marginals. It runs IBSN with `eta=1e-3`, up to 300 outer iterations, and a stopping tolerance of `1e-11`, then rounds the transport plan to the prescribed marginals.

The output includes iteration diagnostics, the row and column marginal residuals, and the transport objective value. No external dataset is needed for this example.

To adapt the solver to your own data, the main calls are:

```python
from Alg.IBSN_Solver import IBSNsolver

# C: cost matrix of shape (m, n)
# a, b: strictly positive marginals of shape (m,) and (n,), each summing to 1
solver = IBSNsolver(C=C, eta=1e-3, a=a, b=b)
X = solver.optimize(max_iter=300, tol=1e-11)
```

Here, `X` is the computed transport plan. Iteration statistics are available in `solver.history`.

## Datasets

The main real-data benchmarks use the following locations:

| Dataset | Location |
| --- | --- |
| MNIST | `data/MNIST/` |
| Fashion-MNIST | `data/FashionMNIST/` |
| DOTmark | `data/DOTmark/` |
 | ImageNet | `data/ImageNet/` |
| Earth's population density | `data/meta-ot/pop-15min.tif` |

Check that the required data files are present before running a real-data experiment. Dataset choices, image indices, and problem sizes are configured in the experiment scripts.

## Running Experiments

Run scripts from the repository root so that imports and relative data paths resolve correctly. Most experiment settings are defined directly in the scripts, including problem sizes, algorithm selections, regularization parameters, tolerances, and output directories.

| Experiment | Scripts                                                     |
| --- |-------------------------------------------------------------|
| Synthetic OT benchmarks | `synthetic.py`, `synthetic_plot.py`                         |
| Real-data OT benchmarks | `RealData.py`, `ImageNet_experiment.py`, `RealData_plot.py` |
| Effect of the regularization parameter $\eta$ | `eta_impact_synthetic.py`, `eta_impact_real.py`             |
| Effect of sparsity in the Newton phase | `sparsity_impact.py`                                        |
| Balanced-size EOT comparisons | `EOT_balanced.py`, `EOT_balanced_plot.py`                   |
| Population-density EOT comparisons with unequal support sizes | `EOT_Unbalanced.py`, `EOT_Unbalanced_plot.py`               |
| EOT sparsity experiments | `EOT_sparsity.py`, `EOT_sparsity_plot.py`                   |
| Color transfer | `Color_transfer/Color_Transfer.py`                          |

For example, after reviewing the settings in the scripts, run a synthetic benchmark and generate its plots:

```bash
python synthetic.py
python synthetic_plot.py
```

Make sure the plotting script uses the same problem settings and result directory as the experiment. Results are generally saved under `Results/`; inspect the output paths before rerunning an experiment to avoid replacing results you want to keep. Some benchmarks compute and cache a reference solution with Gurobi when no cache is available.
