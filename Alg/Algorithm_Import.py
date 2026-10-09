from .GurobiSolver import *
from .HOT_Solver import HOTSolver
from .PINS_solver import PINSsolver
from .IPOT_Solver import IPOTSolver
from .ExtraGrad_Solver import ExtraGrad
from .SSNS_Solver import SSNSSolver
from .SPLR_Solver import SPLRSolver
from .IBSN_Solver import IBSNsolver
from .HPR_Solver import HPRSolver
from .DualExP_Solver import DualExPSolver
from .sGSADMM_Solver import sGSADMMSolver
from .IBSink_Solver import IBSinkSolver
from .Sinkhorn_Solver import SinkhornSolver
import numpy as np
import pandas as pd
import os

def run_IBSN(C, a, b, eta, tol, time_max, opt):
    solver = IBSNsolver(C, eta, a, b, opt)
    X = solver.optimize(max_iter=300, tol=1e-11, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_ssns(C, a, b, eta, tol, time_max, opt):
    solver = SSNSSolver(C=C, eta=eta, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    X = solver.optimize(tol=tol, max_iter=700, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_splr(C, a, b, eta, tol, time_max, opt):
    solver = SPLRSolver(C=C, eta=eta, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    X = solver.optimize(tol=tol, max_iter=700, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_hpr(C, a, b, eta, tol, time_max, opt):
    if C.shape[0] == 1000: max_iter = 5000
    elif C.shape[0] == 5000: max_iter = 2000
    elif C.shape[0] == 10000: max_iter = 1500
    else: max_iter = 1000
    solver = HPRSolver(C=C, eta=eta, a=a, b=b, obj_truth=opt, sigma=None, check_freq=50, adjust_sigma=True)
    X = solver.optimize(tol=tol, max_iter=max_iter, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_dualexp(C, a, b, eta, tol, time_max, opt):
    d_inf = np.linalg.norm(C.reshape(-1), ord=np.inf)
    solver = DualExPSolver(
        C=C,
        eta=eta,
        a=a,
        b=b,
        obj_truth=opt,
        entropy_weight=3.0,
        step_size=d_inf,
        inner_movement_tol=1e-7,
        inner_max_iter=20,
    )
    if C.shape[0] == 1000: max_iter = 100
    elif C.shape[0] == 5000: max_iter = 2000
    elif C.shape[0] == 10000: max_iter = 55
    else: max_iter = 1000
    X = solver.optimize(tol=tol, max_iter=max_iter, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_sgsadmm(C, a, b, eta, tol, time_max, opt):
    solver = sGSADMMSolver(
        C=C,
        eta=eta,
        a=a,
        b=b,
        obj_truth=opt,
        beta=1.0,
        tau=1.0,
        check_freq=50,
        adaptive_beta=False,
    )
    if C.shape[0] == 1000: max_iter = 5000
    elif C.shape[0] == 5000: max_iter = 2000
    elif C.shape[0] == 10000: max_iter = 1500
    else: max_iter = 1000
    X = solver.optimize(max_iter=max_iter, tol=tol, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_Sinkhorn(C, a, b, eta, tol, time_max, opt):
    solver = SinkhornSolver(C, eta, a=a, b=b, obj_truth=opt)
    X = solver.sinkhorn_knopp(numItermax=30000, stopThr=tol, verbose=True)
    return X, solver.history, solver.eta

# def run_ssns(C, a, b, eta, tol, time_max, opt):
#     solver = SSNSSolver(C=C, eta=eta/10, a=a, b=b, obj_truth=opt)
#     X = solver.optimize(tol=1e-9, max_iter=700, time_max=time_max, verbose=True)
#     return X, solver.history, solver.eta
#
# def run_splr(C, a, b, eta, tol, time_max, opt):
#     solver = SPLRSolver(C=C, eta=eta/10, a=a, b=b, obj_truth=opt)
#     X = solver.optimize(tol=1e-9, max_iter=700, time_max=time_max, verbose=True)
#     return X, solver.history, solver.eta

def run_extragradient(C, a, b, eta, tol, time_max, opt):
    solver = ExtraGrad(C, eta, a, b, obj_truth=opt)
    X = solver.optimize(tol=1e-7, max_iter=2000, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_hot_halpern(C, a, b, eta, tol, time_max, opt):
    n = len(a)
    c1 = np.zeros(n, dtype=float)  # 这里就是长度 n 的零向量
    c2 = C.flatten(order='F')  # 与 demo.py 保持相同的列优先展平
    c = np.concatenate([c1, c2])
    c_Max = (c2.max() if c2.size else 1.0) + (c1.max() if c1.size else 0.0)
    if c_Max == 0:
        c_Max = 1.0
    c = c / c_Max

    solver = HOTSolver(
        a_seq=a,
        b_seq=b,
        c=c,
        m=1,
        n=n,
        cMax=c_Max,
        max_iters=20_000,
        tolerance=1e-11,
        check_freq=100,
        sigma=None,
        adjust_sigma=True,
        logging=True,
        dtype=None,  # default float64
        device=None,
        obj_truth=opt
    )
    X = solver.optimize(time_max=time_max, verbose=True)
    return X, solver.history, None


def run_PINS(C, a, b, eta, tol, time_max, opt):
    solver = PINSsolver(C, eta, a=a, b=b, obj_truth=opt)
    X = solver.optimize(maxit=150, tol=1e-11, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta


def run_IPOT(C, a, b, eta, tol, time_max, opt):
    solver = IPOTSolver(C, eta=1e-1, a=a, b=b, obj_truth=opt)
    X = solver.optimize(maxit=4000, tol=1e-11, time_max=time_max, verbose=True)
    return X, solver.history, solver.eta

def run_IBSink(C, a, b, eta, tol, time_max, opt):
    solver = IBSinkSolver(C=C, eta=eta, a=a, b=b, obj_truth=opt)
    if eta == 1e-2: max_iter = 600
    elif eta == 1e-3: max_iter = 300
    elif eta == 1e-4: max_iter = 80
    else: max_iter = 300
    X = solver.optimize(tol=1e-11, max_iter=max_iter)
    return X, solver.history, solver.eta

def load_results(result_root):
    data = {}
    files = sorted(os.listdir(result_root))[::-1]
    for f in files:
        csv_file = os.path.join(result_root, f)
        algo, ext = os.path.splitext(f)
        if ext == '.csv':
            dfs = [pd.read_csv(csv_file)]
            data[algo] = dfs
    return data
