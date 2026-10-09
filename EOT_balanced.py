from Alg.Algorithm_Import import *
from Data_Sample_Import import *
from Alg.EOT_solver import IBSNsolver_EOT, PINSsolver_EOT
import os
import pandas as pd
import numpy as np
import ast
import ot
import timeit

def run_PINS_EOT(C, a, b, eta, tol, time_max, opt):
    solver = PINSsolver_EOT(C, eta, a=a, b=b, obj_truth=opt)
    X = solver.optimize(tol=tol)
    return X, solver.history, solver.eta

def run_IBSN_EOT_skip1(C, a, b, eta, tol, time_max, opt):
    solver = IBSNsolver_EOT(C, eta, a=a, b=b, obj_truth=opt, skip_first_stage=True)
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

time_max = 10000.0
eta = 1e-4
tol = 1e-9

exp = "synthetic"  # "synthetic" or "real"

ID = {
    "DOTmark": {1: "Shapes",
                2: "ClassicImages",
                3: "MicroscopyImages",
                4: "WhiteNoise",
                }
}

# experiment settings
if exp == "synthetic":
    m = n = 1000
    cost_matrix_norm = "Square"  # "Square" or "Uniform"
elif exp == "real":
    this_dir = os.path.dirname(__file__)
    dataset_name = "DOTmark"  # FashionMNIST, DOTmark
    default_dir = os.path.join(this_dir, "data", dataset_name)
    experiment_id = 4  # 1,2,4
    size = 64  # only for DOTmark
    if dataset_name == "FashionMNIST" or dataset_name == "MNIST":
        pid, qid = ID[dataset_name][experiment_id]
        C, a, b = load_mnist(default_dir, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
        RESULT_ROOT = f"Results/cg/Real/{dataset_name}_pid{pid}_qid{qid}"
    elif dataset_name == "DOTmark":
        category = ID[dataset_name][experiment_id]
        pid, qid = 2, 4
        C, a, b = load_dotmark(default_dir, category, size, p_id=pid, q_id=qid, norm_type='euclidean', eps=0.01)
        RESULT_ROOT = f"Results/cg/Real/{dataset_name}_{category}_size{size}"
else:
    raise ValueError(f"Unknown")


if exp == "synthetic":
    seed = 41
    rng = np.random.default_rng(seed)
    RESULT_ROOT = f"Results/cg/Synthetic/cg_compare_m{m}_n{n}"
    print(f"Result root is {RESULT_ROOT}")
    RESULT_ROOT = os.path.join(RESULT_ROOT, str(cost_matrix_norm))
    os.makedirs(RESULT_ROOT, exist_ok=True)

    # ===== cost matrix =====
    if cost_matrix_norm == "Square":
        K = np.arange(m)
        J = np.arange(n)
        C = (J[None, :] - K[:, None]) ** 2
        C = C / np.max(C)
    elif cost_matrix_norm == "Uniform":
        np.random.seed(seed)
        C = np.random.uniform(0, 1, size=(m, n))
        C = C / np.max(C)
    else:
        raise ValueError(f"Unknown cost_matrix_norm {cost_matrix_norm}")

    a = rng.random(m); a /= a.sum()
    b = rng.random(n); b /= b.sum()
opt = None


ALGORITHMS = {
    "SSNS": run_ssns,
    "SPLR": run_splr,
    "(E)PINS": run_PINS_EOT,
    # "IBSN_EOT_skip1": run_IBSN_EOT_skip1,
    "(E)IBSN": run_IBSN_EOT,
    "Sinkhorn": run_Sinkhorn,
}

for alg_name, fn in ALGORITHMS.items():
    DATA_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
    if not os.path.exists(DATA_path):
        print(f" Running {alg_name}...")
        X, history, eta_val = fn(C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
        df = pd.DataFrame({
            "algo": alg_name,
            "m": int(m), "n": int(n),
            "iter": np.arange(len(history['time']), dtype=int),
            "opt": [opt] * len(history.get("time", [])),
            "eta": [eta_val] * len(history.get("time", [])),
        })
        for key in history:
            if not history[key]:
                history[key] = [None] * len(history['time'])
        df = pd.concat([df, pd.DataFrame(history)], axis=1)

        out_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")


loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

def cg_iter_stats_ent(algs):
    alg = loaded[algs][0]
    cg_iter = alg["cg_iter"][2:].to_numpy()
    Ite = alg["iter"].to_numpy()[-1]
    total_time = alg["time"].to_numpy()[-1]
    grad_norm = alg["grad_norm"].to_numpy()[-1]
    # mean and std
    return cg_iter.mean(), cg_iter.std(), Ite, total_time, grad_norm


for a in ALGORITHMS.keys():
    if a in loaded:
        if a != "Sinkhorn":
            cgiter_mean, cgiter_std, Ite, total_time, grad_norm = cg_iter_stats_ent(a)
            print(f"{a}: per CG iters iter = {cgiter_mean:.2f} ± {cgiter_std:.2f} "
                f"with total iters = {Ite} and time = {total_time:.2f}")

# for latex
for a in ALGORITHMS.keys():
    if a in loaded and a != "Sinkhorn":
        cgiter_mean, cgiter_std, Ite, total_time, grad_norm = cg_iter_stats_ent(a)
        if grad_norm > 1e-9:
            cgiter_mean = cgiter_std  = "-"
            print(f"{a} & {cgiter_mean} ({cgiter_std}) & {Ite} & {total_time:.2f}")
        else:
            print(f"{a} & {cgiter_mean:.1f} ({cgiter_std:.0f}) & {Ite} & {total_time:.2f}")

