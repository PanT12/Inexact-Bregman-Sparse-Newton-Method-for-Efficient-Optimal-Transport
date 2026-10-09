import numpy as np
from Alg.base import *
import timeit
import ot


class IPOTSolver(optimal_transport):
    def __init__(self, *args, L=1, sinkhorn_method="Sinkhorn", **kwargs):
        super().__init__(*args, **kwargs)
        self.L = int(L)
        self.sinkhorn_method = sinkhorn_method

    def _safe_div(self, num, den):
        return num / np.maximum(den, 1e-300)

    def optimize(self, maxit=10000, tol=1e-4, time_max=np.inf, verbose=False):
        # initialization
        X = np.outer(self.a, self.b)
        f = np.zeros(self.m)
        g = np.zeros(self.n)
        kkt_err = self.kkt_err(X, self.eta * g, self.eta * f)

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=[], gnorm=None, X=X, newton_iter=0, cg_time=[], first_stage_iter=0)

        for it in range(maxit):
            t_start = timeit.default_timer()

            C_iter = self.C - self.eta * np.log(X)
            warmstart = [f, g]
            X, log = ot.sinkhorn(
                self.a, self.b, C_iter, self.eta, method=self.sinkhorn_method,
                numItermax=self.L, stopThr=1e-11, log=True, warmstart=warmstart
            )
            f, g = np.log(log['u']), np.log(log['v'])
            t_end = timeit.default_timer()
            t_total += (t_end - t_start)

            row_sum, col_sum = X.sum(axis=1), X.sum(axis=0)
            row_diff, col_diff = row_sum - self.a, col_sum - self.b
            diff = np.concatenate([row_diff, col_diff])
            gnorm = np.linalg.norm(diff)
            self.record(t_total=t_total, cg_iter=[], gnorm=gnorm, X=X, newton_iter=0, cg_time=[], first_stage_iter=1)
            kkt_err = self.kkt_err(X, self.eta * g, self.eta * f)

            if verbose and it % 100 == 0:
                print(
                    f"iter = {it:5d}, gnorm = {gnorm: 2.1e}, "
                    f"kkt_err = {kkt_err:2.1e}, time_iter = {t_total:3.2f}"
                )

            if kkt_err < tol:
                break

        return X

if __name__ == "__main__":
    from GurobiSolver import *
    import matplotlib.pyplot as plt
    import os
    m, n = 5, 6
    eta = 1e-1
    opt = None
    # np.random.seed(4)
    a = np.random.rand(m)
    a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # print("a is ", a)
    b = np.random.rand(n)
    b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # print("b is ", b)
    C = np.random.rand(m, n)

    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()
    opt = np.sum(C * gt)

    # m = n = 1000
    # cost_matrix_norm = "Uniform"  # "Square", "Uniform"
    # eta = 1e-4
    # seed = 41
    # RESULT_ROOT = f"../Results/synthetic/alg_compare_m{m}_n{n}_rng{seed}"
    # RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)
    # opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")
    # cache = np.load(opt_cache_path, allow_pickle=False)
    # a = cache["a"]
    # b = cache["b"]
    # opt = float(cache["opt"])
    #
    # # ===== cost matrix =====
    # if cost_matrix_norm == "Square":
    #     J = np.arange(n)
    #     C = (J[None, :] - J[:, None]) ** 2
    #     C = C / np.max(C)
    # elif cost_matrix_norm == "Uniform":
    #     np.random.seed(seed)
    #     C = np.random.uniform(0, 1, size=(m, n))
    #     C = C / np.max(C)

    solver = IPOTSolver(C=C, eta=1.0, a=a, b=b, obj_truth=opt)
    X = solver.optimize(maxit=3000, tol=1e-11, verbose=True)
    X = solver._round_to_marginals(X, a, b)

    print("row difference is ", np.linalg.norm(X.sum(axis=1) - a))
    print("col difference is ", np.linalg.norm(X.sum(axis=0) - b))
    cost = np.sum(C * X)
    print("objective value is ", cost)
    print("difference between cost is ", np.abs(cost - opt))
