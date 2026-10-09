import ot
import time
from Alg.Algorithm_Import import *

m = 800; n = 801
cost_matrix_norm = "Uniform"  # "Square" or "Uniform"
time_max = 10000.0
eta = 1e-4
tol = 1e-7
seed = 41
rng = np.random.default_rng(seed)
RESULT_ROOT = f"Results/synthetic/alg_compare_m{m}_n{n}_rng{seed}"
RESULT_ROOT = os.path.join(RESULT_ROOT, str(cost_matrix_norm))
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


ALGORITHMS = {
    "IBSN": run_IBSN,
    # "ExtraGrad": run_extragradient,
    # "HOT": run_hot_halpern,
    # "PINS": run_PINS,
    # "IBSink": run_IBSink,
    # "IPOT": run_IPOT,
    # "HPR": run_hpr,
    # "sGSADMM": run_sgsadmm,
}

opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

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

time_start = time.time()
res = ot.emd(a, b, C)
time_end = time.time()
print(time_end - time_start)

for alg_name, fn in ALGORITHMS.items():
    print(f" Running {alg_name}...")
    if alg_name == "IBSink":
        X, history, eta_val = fn(C, a, b, eta=1e-3, tol=tol, time_max=time_max, opt=opt)
    else:
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

    # out_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
    # df.to_csv(out_path, index=False)
    # print(f"  saved -> {out_path}")
