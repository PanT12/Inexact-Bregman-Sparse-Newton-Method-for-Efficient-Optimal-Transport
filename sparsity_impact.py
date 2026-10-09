from Alg.Algorithm_Import import *
import pandas as pd
import os
from itertools import chain
import ast


m = n = 1000
cost_matrix_norm = "Square"  # "Square", "Uniform"
eta = 1e-4
time_max = 6000.0
tol = 1e-7
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/sparsity_impact/rho_compare_m{m}_n{n}"
RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)
os.makedirs(RESULT_ROOT, exist_ok=True)

# ===== cost matrix =====
if cost_matrix_norm == "Square":
    J = np.arange(n)
    C = (J[None, :] - J[:, None]) ** 2
    C = C / np.max(C)
elif cost_matrix_norm == "Uniform":
    np.random.seed(seed)
    C = np.random.uniform(0, 1, size=(m, n))
    C = C / np.max(C)
else:
    raise ValueError(f"Unknown cost_matrix_norm: {cost_matrix_norm}")

rho_list = [0.0, None]

opt_cache_path = os.path.join(RESULT_ROOT, f"opt_ab.npz")

if os.path.exists(opt_cache_path):
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    opt = float(cache["opt"])
    print(f"  [cache] loaded a,b,opt from {opt_cache_path}")
else:
    a = rng.random(m); a /= a.sum()
    b = rng.random(n); b /= b.sum()

    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()

    print("row difference is ", np.linalg.norm(a - gt.sum(axis=1)))
    print("col difference is ", np.linalg.norm(b - gt.sum(axis=0)))
    opt = float(np.sum(C * gt))
    print("  [Gurobi] computed opt =", opt)

    np.savez(opt_cache_path, a=a, b=b, opt=opt)
    print(f"  [cache] saved a,b,opt to {opt_cache_path}")

for rho in rho_list:
    DATA_path = os.path.join(RESULT_ROOT, f"{rho}.csv")
    if not os.path.exists(DATA_path):
        print(f" Running rho={rho}...")
        solver = IBSNsolver(C, eta, a, b, opt)
        X = solver.optimize(max_iter=300, tol=1e-11, time_max=time_max, rho=rho, verbose=True)
        history = solver.history
        df = pd.DataFrame({
            "algo": "our",
            "m": int(m), "n": int(n),
            "iter": np.arange(len(history['time']), dtype=int),
            "opt": [opt] * len(history.get("time", [])),
            "eta": [1e-4] * len(history.get("time", [])),
            "rho": [rho] * len(history.get("time", [])),
        })
        for key in history:
            if not history[key]:
                history[key] = [None] * len(history['time'])
        df = pd.concat([df, pd.DataFrame(history)], axis=1)

        out_path = os.path.join(RESULT_ROOT, f"{str(rho)}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")

loaded = load_results(RESULT_ROOT)
print(f"Loaded results for algorithms: {list(loaded.keys())}")

def cg_iter_stats_ori(algs):
    alg = loaded[algs][0]
    cg_time = np.array(list(chain.from_iterable(ast.literal_eval(s) for s in alg["cg_time"][1:])))
    total_iter = len(alg["time"].to_numpy())
    total_time = alg["time"].to_numpy()[-1]
    newton_iter = alg["newton_iter"][1:].to_numpy()
    abs_err = alg["abs_err"].to_numpy()[-1]
    return cg_time.sum(), total_iter, newton_iter.sum(), total_time, abs_err


for a in rho_list:
    if str(a) in loaded:
        cg_time, total_iter, newton_iter, total_time, gap = cg_iter_stats_ori(str(a))
        print(f"{a}: total CG time = {cg_time:.2f} seconds")
        print(f"{a}: total outer iters = {total_iter}")
        print(f"{a}: total Newton iters = {newton_iter}")
        print(f"{a}: total time = {total_time:.2f} seconds")
        print(f"{a}: final gap = {gap:.2e}")

# for latex
for a in rho_list:
    if str(a) in loaded:
        cg_time, _, newton_iter, total_time, gap = cg_iter_stats_ori(str(a))
        if str(a) == "0.0": a = "IBN"
        elif str(a) == "None": a = "IBSN"
        print(f"{a} & {newton_iter} & {cg_time:.2f} & {total_time:.2f} & {gap:.2e}")
