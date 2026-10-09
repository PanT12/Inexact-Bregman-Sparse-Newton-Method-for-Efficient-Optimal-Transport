import timeit
import warnings

import numpy as np

from Alg.base import *


class sGSADMMSolver(optimal_transport):
    """
    OT-specialized sGS-ADMM adapted from Algorithm 1 in
    "A Fast Globally Linearly Convergent Algorithm for the Computation of Wasserstein Barycenters".

    We solve the dual LP of OT in the form
        min  delta_+(V) - <a, y> - <b, z>
        s.t. V - C + y 1^T + 1 z^T = 0.

    The update order follows the paper's symmetric Gauss-Seidel pattern:
        Step 1:  V
        Step 2a: z_tilde
        Step 2b: y
        Step 2c: z
        Step 3:  multiplier
    """

    def __init__(
        self,
        *args,
        beta=1.0,
        tau=1.0,
        check_freq=25,
        adaptive_beta=False,
        beta_min=1e-4,
        beta_max=1e4,
        change_tol=1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self.tau = float(tau)
        self.check_freq = int(check_freq)
        self.adaptive_beta = adaptive_beta
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.change_tol = float(change_tol)

        self.history["kkt_err"] = []
        self.history["beta"] = []
        self.history["dual_gap"] = []
        self.history["primal_res"] = []
        self.history["dual_res"] = []
        self.history["successive_change"] = []

    def _scaled_cost(self):
        scale = np.linalg.norm(self.C, ord="fro")
        if scale <= 0:
            scale = 1.0
        return self.C / scale, scale

    def _paper_residuals(self, V, y, z, Lambda, D):
        v_norm = np.linalg.norm(V, ord="fro")
        lambda_norm = np.linalg.norm(Lambda, ord="fro")
        d_norm = np.linalg.norm(D, ord="fro")
        y_norm = np.linalg.norm(y)
        z_norm = np.linalg.norm(z)
        ones_m = np.ones(self.m)
        ones_n = np.ones(self.n)

        eta1 = np.linalg.norm(V - np.maximum(V - Lambda, 0.0), ord="fro")
        eta1 /= (1.0 + v_norm + lambda_norm)

        eta2 = np.linalg.norm(Lambda @ ones_n - self.a)
        eta2 /= (1.0 + np.linalg.norm(self.a) + lambda_norm)

        eta3 = np.linalg.norm(Lambda.T @ ones_m - self.b)
        eta3 /= (1.0 + np.linalg.norm(self.b) + lambda_norm)

        eta4 = np.linalg.norm(V - D + y[:, None] + z[None, :], ord="fro")
        eta4 /= (1.0 + d_norm + v_norm + y_norm + z_norm)

        eta5 = np.linalg.norm(np.minimum(Lambda, 0.0), ord="fro")
        eta5 /= (1.0 + lambda_norm)

        primal_res = max(0.7 * eta1, eta2, eta3, eta5)
        dual_res = eta4

        X = np.maximum(Lambda, 0.0)
        obj_p = float(np.sum(D * X))
        obj_d = float(self.a @ y + self.b @ z)
        dual_gap = abs(obj_p - obj_d) / (1.0 + abs(obj_p) + abs(obj_d))

        return primal_res, dual_res, dual_gap

    def _update_beta(self, primal_res, dual_res):
        if primal_res <= 0 and dual_res <= 0:
            return

        if primal_res <= 0:
            ratio = np.inf
        else:
            ratio = dual_res / primal_res

        inv_ratio = np.inf if (not np.isfinite(ratio) or ratio <= 0) else 1.0 / ratio
        max_ratio = max(ratio if np.isfinite(ratio) else np.inf, inv_ratio if np.isfinite(inv_ratio) else np.inf)

        if max_ratio <= 10:
            sigma = 1.05
        elif max_ratio > 100:
            sigma = 1.5
        else:
            sigma = 1.2

        if ratio > 2.0:
            self.beta = min(self.beta * sigma, self.beta_max)
        elif inv_ratio > 2.0:
            self.beta = max(self.beta / sigma, self.beta_min)

    def _record_iter(self, t_total, X, primal_res, dual_res, dual_gap, successive_change):
        self.record(
            t_total=t_total,
            cg_iter=[],
            gnorm=None,
            X=X,
            newton_iter=len(self.history["time"]),
            cg_time=[],
            first_stage_iter=0,
        )
        kkt_err = max(primal_res, dual_res, dual_gap)
        self.history["kkt_err"].append(kkt_err)
        self.history["beta"].append(self.beta)
        self.history["dual_gap"].append(dual_gap)
        self.history["primal_res"].append(primal_res)
        self.history["dual_res"].append(dual_res)
        self.history["successive_change"].append(successive_change)

    def optimize(self, max_iter, tol, time_max=np.inf, verbose=True):
        D, _ = self._scaled_cost()
        ones_m = np.ones(self.m)
        ones_n = np.ones(self.n)

        y = np.zeros(self.m, dtype=float)
        z = np.zeros(self.n, dtype=float)
        V = np.zeros((self.m, self.n), dtype=float)
        Lambda = np.zeros((self.m, self.n), dtype=float)

        t_total = 0.0
        primal_res, dual_res, dual_gap = self._paper_residuals(V, y, z, Lambda, D)
        X = np.maximum(Lambda, 0.0)
        self._record_iter(t_total, X, primal_res, dual_res, dual_gap, np.inf)

        if verbose:
            print(
                f"iter = {0:5d}, kkt_err = {self.history['kkt_err'][-1]:2.2e}, "
                f"dual_gap = {dual_gap:2.2e}, beta = {self.beta:2.2e}, time = {t_total:3.2f}"
            )

        for k in range(max_iter):
            t_start = timeit.default_timer()
            Lambda_old = Lambda.copy()
            y_old = y.copy()
            z_old = z.copy()

            # Step 1: V update
            Q = D - y_old[:, None] - z_old[None, :] - Lambda / self.beta
            V = np.maximum(Q, 0.0)
            B = V - D - Lambda / self.beta

            # Step 2a: z-tilde update
            z_tilde = -((V - D + y_old[:, None]).T @ ones_m) / self.m
            z_tilde -= (Lambda.T @ ones_m - self.b) / (self.beta * self.m)

            # Step 2b: y update
            y = -((V - D + z_tilde[None, :]) @ ones_n) / self.n
            y -= (Lambda @ ones_n - self.a) / (self.beta * self.n)

            # Step 2c: z update
            z = -((V - D + y[:, None]).T @ ones_m) / self.m
            z -= (Lambda.T @ ones_m - self.b) / (self.beta * self.m)

            # Step 3: multiplier update
            residual = V - D + y[:, None] + z[None, :]
            Lambda = Lambda + self.tau * self.beta * residual

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            X = np.maximum(Lambda, 0.0)
            primal_res, dual_res, dual_gap = self._paper_residuals(V, y, z, Lambda, D)
            successive_change = (
                np.linalg.norm(Lambda - Lambda_old, ord="fro") / (1.0 + np.linalg.norm(Lambda_old, ord="fro"))
                + np.linalg.norm(y - y_old) / (1.0 + np.linalg.norm(y_old))
                + np.linalg.norm(z - z_old) / (1.0 + np.linalg.norm(z_old))
            )

            if self.adaptive_beta and (k + 1) % self.check_freq == 0:
                self._update_beta(primal_res, dual_res)

            if verbose and ((k + 1) % self.check_freq == 0 or k == 0):
                self._record_iter(t_total, X, primal_res, dual_res, dual_gap, successive_change)
                print(
                    f"iter = {k + 1:5d}, kkt_err = {self.history['kkt_err'][-1]:2.2e}, "
                    f"dual_gap = {dual_gap:2.2e}, beta = {self.beta:2.2e}, time = {t_total:3.2f}"
                )

            if max(primal_res, dual_res, dual_gap) < tol:
                break
            if successive_change < self.change_tol and max(primal_res, dual_res) < max(1e-3, 10 * tol):
                break
            if t_total > time_max:
                break
        else:
            warnings.warn("sGS-ADMM reached max_iter before satisfying the stopping rule.")

        return self._round_to_marginals(np.maximum(Lambda, 0.0), self.a, self.b)
