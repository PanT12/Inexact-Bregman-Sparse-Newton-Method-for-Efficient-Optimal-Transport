from Data_Sample_Import import *
from Alg.Algorithm_Import import *
import numpy as np
import os
import pandas as pd
import ot
import argparse


ALGORITHMS = {
    "IBSN": run_IBSN,
    "HOT": run_hot_halpern,
    "PINS": run_PINS,
    "IBSink": run_IBSink,
    # "IPOT": run_IPOT,
    # "ExtraGrad": run_extragradient,
}


parser = argparse.ArgumentParser(
    description="Run OT experiments on ImageNet-style image folders."
)
parser.add_argument("--p-id", type=int, default=333, help="Source image index after sorting paths.")
parser.add_argument("--q-id", type=int, default=12580, help="Target image index after sorting paths.")
parser.add_argument("--image-size", type=int, default=100, help="Resize images to image-size x image-size.")
args = parser.parse_args()


# Images are recursively loaded from data_dir, sorted by path.
p_id, q_id = args.p_id, args.q_id
image_size = args.image_size
dataset_name = 'ImageNet'

this_dir = os.path.dirname(__file__)
default_dir = os.path.join(this_dir, "data", dataset_name)

C, a, b, p_path, q_path = load_imagenet(
    default_dir,
    p_id=p_id,
    q_id=q_id,
    image_size=image_size,
    norm_type='euclidean',
    eps=0.01,
)

RESULT_ROOT = f"Results/real/{dataset_name}/pid{p_id}_qid{q_id}"
time_max = 4000.0
eta = 1e-3
tol = 1e-7
os.makedirs(RESULT_ROOT, exist_ok=True)

print("source image:", p_path)
print("target image:", q_path)
print("a shape:", a.shape)
print("b shape:", b.shape)
print("C shape:", C.shape)
print("maximum value in a:", np.max(a))
print("maximum value in b:", np.max(b))


opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")

if os.path.exists(opt_cache_path):
    cache = np.load(opt_cache_path, allow_pickle=False)
    opt = float(cache["opt"])
    print(f"  [cache] loaded opt from {opt_cache_path}")
else:
    print("  [POT] running ot.emd for reference optimum...")
    gt = ot.emd(a, b, C, numItermax=1000000)

    print("row difference is ", np.linalg.norm(a - gt.sum(axis=1)))
    print("col difference is ", np.linalg.norm(b - gt.sum(axis=0)))
    opt = float(np.sum(C * gt))
    print("  [POT] computed opt =", opt)

    np.savez(opt_cache_path, opt=opt, p_path=p_path, q_path=q_path)
    print(f"  [cache] saved opt to {opt_cache_path}")


for alg_name, fn in ALGORITHMS.items():
    out_path = os.path.join(RESULT_ROOT, f"{alg_name}.csv")
    if os.path.exists(out_path):
        print(f" Skipping {alg_name}: existing result found at {out_path}")
        continue

    print(f" Running {alg_name}...")
    X, history, eta_val = fn(C, a, b, eta=eta, tol=tol, time_max=time_max, opt=opt)
    df = pd.DataFrame({
        "algo": alg_name,
        "data_name": dataset_name,
        "iter": np.arange(len(history['time']), dtype=int),
        "opt": [opt] * len(history.get("time", [])),
        "eta": [eta_val] * len(history.get("time", [])),
    })
    for key in history:
        if not history[key]:
            history[key] = [None] * len(history['time'])
    df = pd.concat([df, pd.DataFrame(history)], axis=1)

    df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
