import math
import timeit

import numpy as np

from Alg.base import *


class HPRSolver(optimal_transport):
    """
    HPR method specialized from Algorithm 5 in
    "An Efficient HPR Algorithm for the Wasserstein Barycenter Problem
    with O(Dim(P):epsilon) Computational Complexity".

    This implementation targets the OT LP:
        min <C, P>
        s.t. P 1 = a, P^T 1 = b, P >= 0.

    We eliminate the last column equality so that A A^* is nonsingular.
    """

    def __init__(self, *args, sigma=None, check_freq=50, adjust_sigma=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.check_freq = check_freq
        self.adjust_sigma = adjust_sigma
        self.inner_count = 0
        self.num_restart = 0
        self.kkt_res_old = np.inf
        self.kkt_base = None

        self.history["kkt_err"] = []

    def _pack_plan(self, X):
        return np.asarray(X, dtype=float).reshape(-1, order="C")

    def _unpack_plan(self, x):
        return np.asarray(x, dtype=float).reshape(self.m, self.n, order="C")

    def _apply_A(self, x):
        X = self._unpack_plan(x)
        row_sum = X.sum(axis=1)
        col_sum_tilde = X[:, :-1].sum(axis=0)
        return np.concatenate([row_sum, col_sum_tilde])

    def _apply_AT(self, y):
        alpha = y[:self.m]
        beta_tilde = y[self.m:]
        ATy = np.repeat(alpha[:, None], self.n, axis=1)
        if self.n > 1:
            ATy[:, :-1] += beta_tilde[None, :]
        return ATy.reshape(-1, order="C")

    def _solve_AAT(self, rhs):
        rhs_alpha = rhs[:self.m]
        rhs_beta = rhs[self.m:]

        sum_rhs_alpha = rhs_alpha.sum()
        sum_rhs_beta = rhs_beta.sum()

        sum_alpha = sum_rhs_alpha - sum_rhs_beta
        sum_beta = (sum_rhs_beta - (self.n - 1) * sum_alpha) / self.m

        alpha = (rhs_alpha - sum_beta) / self.n
        beta_tilde = (rhs_beta - sum_alpha) / self.m
        return np.concatenate([alpha, beta_tilde])

    def _full_dual(self, y):
        alpha = y[:self.m]
        beta = np.zeros(self.n, dtype=float)
        if self.n > 1:
            beta[:-1] = y[self.m:]
        return alpha, beta

    def _compute_kkt(self, X, y, z):
        x_vec = self._pack_plan(X)
        primal_res = np.linalg.norm(self._apply_A(x_vec) - np.concatenate([self.a, self.b[:-1]])) / (1.0 + np.linalg.norm(np.concatenate([self.a, self.b[:-1]])))
        dual_res = np.linalg.norm(self._apply_AT(y) + z - self._pack_plan(self.C)) / (1.0 + np.linalg.norm(self._pack_plan(self.C)))
        nonneg_res = np.linalg.norm(np.minimum(x_vec, 0.0)) / (1.0 + np.linalg.norm(x_vec))
        dual_nonneg_res = np.linalg.norm(np.minimum(z, 0.0)) / (1.0 + np.linalg.norm(z))
        compl_res = abs(np.dot(np.maximum(x_vec, 0.0), np.maximum(z, 0.0))) / (1.0 + np.linalg.norm(x_vec) + np.linalg.norm(z))
        return max(primal_res, dual_res, nonneg_res, dual_nonneg_res, compl_res)

    def _initialize_sigma(self, c_vec, b_vec):
        if self.sigma is not None:
            return float(self.sigma)
        c_norm = np.linalg.norm(c_vec)
        b_norm = np.linalg.norm(b_vec)
        if c_norm <= 0:
            return 1.0
        return b_norm / c_norm / math.sqrt(max(self.m, self.n))

    def _record_iter(self, t_total, X, y, z):
        X_nonneg = np.maximum(X, 0.0)
        self.record(
            t_total=t_total,
            cg_iter=[],
            gnorm=None,
            X=X_nonneg,
            newton_iter=self.inner_count,
            cg_time=[],
            first_stage_iter=0,
        )
        kkt = self._compute_kkt(X, y, z)
        self.history["kkt_err"].append(kkt)
        return kkt

    def _restart(self, x_bar, z_bar, current_kkt):
        delta_x = np.linalg.norm(x_bar - self.x_anchor)
        delta_z = np.linalg.norm(z_bar - self.z_anchor)
        if delta_z > 0:
            temp = delta_x * (np.linalg.norm(x_bar) + 1.0) / (delta_z * (np.linalg.norm(z_bar) + 1.0))
            if np.isfinite(temp) and temp > 0:
                self.sigma = float(np.sqrt(temp))

        self.x_anchor = x_bar.copy()
        self.z_anchor = z_bar.copy()
        self.inner_count = 0
        self.num_restart += 1
        self.kkt_base = current_kkt

    def optimize(self, tol, max_iter, time_max=np.inf, verbose=True):
        b_vec = np.concatenate([self.a, self.b[:-1]])
        c_vec = self._pack_plan(self.C)

        X0 = np.outer(self.a, self.b)
        x = self._pack_plan(X0)
        z = np.zeros_like(x)
        x0 = x.copy()
        z0 = z.copy()
        self.x_anchor = x0.copy()
        self.z_anchor = z0.copy()
        y = np.zeros(self.m + self.n - 1, dtype=float)

        sigma = self._initialize_sigma(c_vec, b_vec)
        self.sigma = sigma

        X_bar = self._unpack_plan(x)
        t_total = 0.0
        kkt = self._record_iter(t_total, X_bar, y, z)
        self.kkt_base = kkt
        self.kkt_res_old = kkt

        if verbose:
            print(f"iter = {0:5d}, kkt_err = {kkt:2.2e}, time = {t_total:3.2f}, sigma = {sigma:2.2e}")

        for k in range(max_iter):
            t_start = timeit.default_timer()

            rhs = b_vec / sigma - self._apply_A(x / sigma + z - c_vec)
            y_bar = self._solve_AAT(rhs)
            aty_minus_c = self._apply_AT(y_bar) - c_vec

            x_bar = x + sigma * (z + aty_minus_c)
            z_bar = np.maximum(-(x_bar + sigma * aty_minus_c), 0.0) / sigma

            multiplier = (self.inner_count + 1) / (self.inner_count + 2)
            x = multiplier * (2.0 * x_bar - x) + self.x_anchor / (self.inner_count + 2)
            z = multiplier * (2.0 * z_bar - z) + self.z_anchor / (self.inner_count + 2)

            self.inner_count += 1
            y = y_bar
            X_bar = self._unpack_plan(x_bar)

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            kkt = None
            if (k + 1) % self.check_freq == 0 or k == max_iter - 1:
                kkt = self._record_iter(t_total, X_bar, y, z_bar)

                if self.adjust_sigma:
                    should_restart = False
                    if (k + 1) <= 500:
                        should_restart = True
                    elif kkt < self.kkt_res_old or (k + 1) % 500 == 0:
                        should_restart = True

                    if should_restart:
                        self._restart(x_bar, z_bar, kkt)
                        x = x_bar.copy()
                        z = z_bar.copy()

                self.kkt_res_old = kkt

                if verbose:
                    print(
                        f"iter = {k + 1:5d}, kkt_err = {kkt:2.2e}, time = {t_total:3.2f}, "
                        f"sigma = {self.sigma:2.2e}, restart = {self.num_restart}"
                    )

            if kkt is not None and kkt < tol:
                break
            if t_total > time_max:
                break

        return np.maximum(X_bar, 0.0)


if __name__ == "__main__":
    from GurobiSolver import *
    import matplotlib.pyplot as plt
    import os
    m, n = 200, 200
    eta = 1e-3
    opt = None
    # np.random.seed(1)
    a = np.random.rand(m)
    a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # print("a is ", a)
    b = np.random.rand(n)
    b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # print("b is ", b)
    C = np.random.rand(m, n)

    J = np.arange(n)
    C = (J[None, :] - J[:, None]) ** 2
    C = C / np.max(C)

    guro = Gurobisolver(C, a, b)
    gt = guro.Gurobi_Solver_original()
    opt = np.sum(C * gt)

    # m = n = 5000
    # cost_matrix_norm = "Square"  # "Square", "Uniform"
    # eta = 1e-4 if cost_matrix_norm == "Square" else 1e-3
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

    solver = HPRSolver(C=C, eta=eta, a=a, b=b, obj_truth=opt, sigma=None, check_freq=50, adjust_sigma=True)
    X = solver.optimize(max_iter=20000, tol=1e-11)
    X = solver._round_to_marginals(X, a, b)

    print("row difference is ", np.linalg.norm(X.sum(axis=1) - a))
    print("col difference is ", np.linalg.norm(X.sum(axis=0) - b))
    print("objective value is ", np.sum(C * X))

    plt.plot(solver.history["time"], solver.history["abs_err"])
    plt.yscale('log')
    plt.show()

    cost = np.sum(C * X)

    # print("column sum err is ", np.linalg.norm(X.sum(axis=0) - b))

    # print("difference between gurobi and ssns", np.linalg.norm(X - gt))
    print("difference between cost is ", np.abs(cost - opt))
