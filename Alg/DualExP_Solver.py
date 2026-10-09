import math
import timeit
import warnings

import numpy as np

from Alg.base import *


class DualExPSolver(optimal_transport):
    """
    Dual extrapolation solver based on Algorithm 3 in
    "A Direct O~(1/epsilon) Iteration Parallel Algorithm for Optimal Transport".

    The outer loop follows Algorithm 3. The inner alternating minimization uses the
    practical early-stopping rule from Appendix D: stop when consecutive iterates move
    negligibly in l1.
    """

    def __init__(
        self,
        *args,
        inner_movement_tol=1e-8,
        inner_max_iter=None,
        entropy_weight=3.0,
        step_size=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inner_movement_tol = inner_movement_tol
        self.inner_max_iter = inner_max_iter
        self.entropy_weight = entropy_weight
        self.step_size = step_size
        self.history["kkt_err"] = []
        self.history["inner_iter"] = []
        self.history["dual_gap"] = []

    def _pack_plan(self, X):
        return np.asarray(X, dtype=float).reshape(-1, order="C")

    def _unpack_plan(self, x):
        return np.asarray(x, dtype=float).reshape(self.m, self.n, order="C")

    def _apply_A(self, x):
        X = self._unpack_plan(x)
        row_sum = X.sum(axis=1)
        col_sum = X.sum(axis=0)
        return np.concatenate([row_sum, col_sum])

    def _apply_AT(self, y):
        alpha = y[:self.m]
        beta = y[self.m:]
        return (alpha[:, None] + beta[None, :]).reshape(-1, order="C")

    def _g_x(self, y, d_inf, d_vec):
        return d_vec + 2.0 * d_inf * self._apply_AT(y)

    def _g_y(self, x, d_inf, b_vec):
        return 2.0 * d_inf * (b_vec - self._apply_A(x))

    def _grad_r_x(self, x, y, d_inf):
        x_safe = np.maximum(x, 1e-300)
        return 2.0 * d_inf * (
            self.entropy_weight * (np.log(x_safe) + 1.0) + self._apply_AT(y ** 2)
        )

    def _grad_r_y(self, x, y, d_inf):
        return 4.0 * d_inf * self._apply_A(x) * y

    def _theta_bound(self, d_inf):
        if self.m * self.n <= 1:
            entropy_range = 0.0
        else:
            entropy_range = self.entropy_weight * d_inf * np.log(self.m * self.n)
        return entropy_range + 4.0 * d_inf

    def _theoretical_inner_cap(self, eps, theta, d_inf):
        if self.inner_max_iter is not None:
            return self.inner_max_iter
        eps_safe = max(eps, 1e-15)
        gamma = ((88.0 * d_inf / eps_safe) + 2.0) * theta / (eps_safe ** 2)
        gamma = max(gamma, 1.0 + 1e-15)
        return int(np.ceil(24.0 * np.log(gamma)))

    def _solve_prox_altmin(self, x_ref, y_ref, gx, gy, x_init, y_init, d_inf, eps):
        x_cur = np.asarray(x_init, dtype=float).copy()
        y_cur = np.asarray(y_init, dtype=float).copy()
        inner_cap = self._theoretical_inner_cap(eps, self._theta_bound(d_inf), d_inf)

        if d_inf <= 0:
            return x_cur, y_cur, 0

        grad_rx_ref = self._grad_r_x(x_ref, y_ref, d_inf)
        grad_ry_ref = self._grad_r_y(x_ref, y_ref, d_inf)
        step = self.step_size if self.step_size is not None else d_inf
        coeff_x = step * gx - grad_rx_ref
        coeff_y = step * gy - grad_ry_ref

        for k in range(inner_cap):
            aty_sq = self._apply_AT(y_cur ** 2)
            logits = -(
                coeff_x / (2.0 * d_inf * self.entropy_weight) +
                aty_sq / self.entropy_weight
            )
            logits -= np.max(logits)
            x_next = np.exp(logits)
            x_next /= np.sum(x_next)

            ax = np.maximum(self._apply_A(x_next), 1e-300)
            y_next = -coeff_y / (4.0 * d_inf * ax)
            y_next = np.clip(y_next, -1.0, 1.0)

            movement = np.linalg.norm(x_next - x_cur, 1) + np.linalg.norm(y_next - y_cur, 1)
            x_cur, y_cur = x_next, y_next
            if movement <= self.inner_movement_tol:
                return x_cur, y_cur, k + 1

        return x_cur, y_cur, inner_cap

    def _dual_gap(self, x, y, d_inf, d_vec, b_vec):
        primal_val = d_vec @ x + 2.0 * d_inf * np.linalg.norm(self._apply_A(x) - b_vec, 1)
        dual_coeff = d_vec + 2.0 * d_inf * self._apply_AT(y)
        dual_val = -2.0 * d_inf * (b_vec @ y) + np.min(dual_coeff)
        return float(primal_val - dual_val)

    def _record(self, t_total, X, inner_iter, dual_gap, y):
        self.record(
            t_total=t_total,
            cg_iter=[],
            gnorm=None,
            X=X,
            newton_iter=len(self.history["time"]),
            cg_time=[],
            first_stage_iter=0,
        )
        self.history["inner_iter"].append(inner_iter)
        self.history["dual_gap"].append(dual_gap)
        alpha = y[:self.m]
        beta = y[self.m:]
        self.history["kkt_err"].append(self.kkt_err(np.maximum(X, 0.0), beta, alpha))

    def optimize(self, tol, max_iter, time_max=np.inf, verbose=True):
        X0 = np.ones((self.m, self.n), dtype=float) / (self.m * self.n)
        x_t = self._pack_plan(X0)
        y_t = np.zeros(self.m + self.n, dtype=float)

        d_vec = self._pack_plan(self.C)
        d_inf = np.linalg.norm(d_vec, ord=np.inf)
        b_vec = np.concatenate([self.a, self.b])

        if d_inf <= 0:
            X = np.outer(self.a, self.b)
            self._record(0.0, X, 0, 0.0, np.zeros(self.m + self.n, dtype=float))
            return X

        t_total = 0.0
        dual_gap = self._dual_gap(x_t, y_t, d_inf, d_vec, b_vec)
        self._record(t_total, self._unpack_plan(x_t), 0, dual_gap, y_t)
        x_mid, y_mid = x_t.copy(), y_t.copy()

        if verbose:
            print(f"iter = {0:5d}, dual_gap = {dual_gap:2.2e}, time = {t_total:3.2f}")

        for t in range(max_iter):
            t_start = timeit.default_timer()

            gx_t = self._g_x(y_t, d_inf, d_vec)
            gy_t = self._g_y(x_t, d_inf, b_vec)
            x_mid, y_mid, inner_iter_1 = self._solve_prox_altmin(
                x_t, y_t, gx_t, gy_t, x_t, y_t, d_inf, tol
            )
            gx_mid = self._g_x(y_mid, d_inf, d_vec)
            gy_mid = self._g_y(x_mid, d_inf, b_vec)
            x_next, y_next, inner_iter_2 = self._solve_prox_altmin(
                x_t, y_t, gx_mid, gy_mid, x_mid, y_mid, d_inf, tol
            )

            dual_gap = self._dual_gap(x_mid, y_mid, d_inf, d_vec, b_vec)

            t_end = timeit.default_timer()
            t_total += t_end - t_start


            if verbose and (t % 1 == 0):
                X_current = self._unpack_plan(x_mid)
                self._record(t_total, X_current, inner_iter_1 + inner_iter_2, dual_gap, y_mid)
                print(
                    f"iter = {t + 1:5d}, dual_gap = {dual_gap:2.2e}, "
                    f"grad_norm = {self.history['grad_norm'][-1]:2.2e}, time = {t_total:3.2f}"
                )

            x_t, y_t = x_next, y_next
            if dual_gap <= tol or t_total > time_max:
                break
        else:
            warnings.warn("DualExP reached max_iter before satisfying the stopping rule.")

        X_tilde = self._unpack_plan(x_mid)
        return self._round_to_marginals(X_tilde, self.a, self.b)

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

    solver = DualExPSolver(C=C, eta=1e-2, a=a, b=b, obj_truth=opt)
    X = solver.optimize(max_iter=5000, tol=1e-11)
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
