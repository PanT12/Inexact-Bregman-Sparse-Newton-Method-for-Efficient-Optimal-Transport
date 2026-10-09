import jax
import numpy as np
import os
import sys
import pandas as pd
import timeit
import ot
import matplotlib.pyplot as plt

from Alg.Algorithm_Import import *
from Alg.EOT_solver import IBSNsolver_EOT, PINSsolver_EOT
# from Unbalanced_EOT_plot import *
from meta_ot import data

def run_PINS_EOT(C, a, b, eta, tol, time_max, opt):
    solver = PINSsolver_EOT(C, eta, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta

def run_IBSN_EOT(C, a, b, eta, tol, time_max, opt):
    solver = IBSNsolver_EOT(C, eta, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta

def Sinkhorn_EOT(C, a, b, eta, tol):
    t_start = timeit.default_timer()
    X = ot.sinkhorn(a, b, C, reg=eta, numItermax=10000, stopThr=tol)
    t_end = timeit.default_timer()
    total_time = t_end - t_start
    row_sum, col_sum = X.sum(axis=1), X.sum(axis=0)
    grad = np.concatenate([a - row_sum, b - col_sum])
    grad_norm = np.linalg.norm(grad)
    return X, grad_norm, total_time

def regularize_distribution(mu, eps=0.01):
    mu = np.array(mu, dtype=np.float64).flatten()
    
    if mu.sum() != 0:
        mu = mu / mu.sum()
        
    n = len(mu)
    mu = (1 - eps / 8) * mu + eps / (8 * n)
    
    return mu

def get_fixed_benchmark_instance(seed=41, eps=0.01, n_demand=10000, n_supply=100, normalize_c=True):

    sampler = data.WorldPairSampler(batch_size=1, n_demand=n_demand, n_supply=n_supply, debug=False)

    C = np.array(sampler.geom.cost_matrix, dtype=np.float64)
    if normalize_c:
        C /= np.max(C)
    
    key = jax.random.PRNGKey(seed)
    batch = sampler(key)
    
    a_raw = np.array(batch.a[0]) # (n_supply,) -> m
    b_raw = np.array(batch.b[0]) # (n_demand,) -> n
    a_fixed = regularize_distribution(a_raw, eps=eps)
    b_fixed = regularize_distribution(b_raw, eps=eps)
    
    transposed = False
    
    m = C.shape[0] # a 的长度
    n = C.shape[1] # b 的长度
    
    if m < n:
        print(f"m({m}) < n({n}). To maintain a lower dimension of Hessian matrix({m}x{m}), take transpose.")
        a_fixed, b_fixed = b_fixed, a_fixed
        C = C.T
        transposed = True
    else:
        print(f"m({m}) >= n({n}). No need to transpose.")
        transposed = False
        
    return C, a_fixed, b_fixed, transposed

seed = 41
eps = 0.01
NORMALIZE_C = True

M_LIST = [50, 80]
ratio = [1, 10, 100, 500, 1000, 5000, 10000]
# N_LIST = [m * r for r in ratio for m in M_LIST]

time_max = 10000.0
eta = 1e-3
tol = 1e-9

ALGORITHMS = {
    "(E)IBSN": run_IBSN_EOT,
    "Sinkhorn": run_Sinkhorn,
    "SSNS": run_ssns,
    "SPLR": run_splr,
    "(E)PINS": run_PINS_EOT,
}

for M_FIXED in M_LIST:
    print(f"\n{'#'*60}")
    print(f"Fix m = {M_FIXED}. Starting Earth EOT experiments over n with various ratios...")
    print(f"{'#'*60}")
    N_LIST = [M_FIXED * r for r in ratio]

    for n_demand in N_LIST:
        print(f"\n" + "="*50)
        print(f"Experiment: m={M_FIXED}, n={n_demand} (Ratio :{n_demand/M_FIXED:.1f})")
        print("="*50)

        RESULT_ROOT = f"Results/earth_eot{M_FIXED}_{eta}/earth_{n_demand}"
        os.makedirs(RESULT_ROOT, exist_ok=True)

        C, a, b, transposed = get_fixed_benchmark_instance(
            seed=seed, eps=eps,
            n_demand=n_demand, n_supply=M_FIXED,
            normalize_c=NORMALIZE_C
        )

        # print(f"Data prepared for solver: C shape={C.shape}")
        # print(f"  -> Optimization variable dim (b size): {b.shape[0]}")
        # print(f"  -> Constraint dim (a size): {a.shape[0]}")
        opt = None

        for alg_name, fn in ALGORITHMS.items():
            print(f"  > Running {alg_name}...")
            DATA_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
            if not os.path.exists(DATA_path):
                X, history, eta_val = fn(C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)

                length = len(history.get('time', []))
                df_dict = {
                    "algo": [alg_name] * length,
                    "data_name": f"earth_m{M_FIXED}_n{n_demand}",
                    "iter": np.arange(length, dtype=int),
                    "opt": [opt] * length,
                    "eta": [eta_val] * length,
                    "n_supply": [M_FIXED] * length,
                    "n_demand": [n_demand] * length,
                    "ratio": [n_demand/M_FIXED] * length
                }

                for key, val in history.items():
                    if len(val) == length:
                        df_dict[key] = val
                    else:
                        if len(val) > length:
                            df_dict[key] = val[:length]
                        else:
                            df_dict[key] = val + [None]*(length - len(val))

                df = pd.DataFrame(df_dict)
                out_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
                df.to_csv(out_path, index=False)

print("\n All Earth EOT experiments completed.")

