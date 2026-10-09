import math

import numpy as np

from Alg.base import *
import timeit
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import logsumexp
from math import ceil
import warnings
import ot



class IBSNsolver_EOT(optimal_transport):
    def __init__(self, *args, skip_first_stage=False, first_phase_method="Sinkhorn", **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.argmax = np.argmax(self.a)

        self.skip_first_stage = skip_first_stage
        self.first_phase_method = first_phase_method

        self.gnorm = []
        self.KL = []

    def update_P_X(self, g, fixed_term):
        exp_term = np.exp((g - np.max(g)) / self.eta)
        numerator = fixed_term * exp_term
        row_sum = numerator.sum(axis=1, keepdims=True)
        if np.any(row_sum == 0.0):
            warnings.warn("Updating P: A row in P has all zero entries. It is better to increase regularization parameter eta.")
            zero_indices = np.where(row_sum == 0)[0]
            numerator[zero_indices, :] = np.random.uniform(low=0.0, high=1.0, size=(len(zero_indices), self.n))
            row_sum = numerator.sum(axis=1, keepdims=True)

        P = numerator / row_sum
        newX = P * self.a[:, None]
        return P, newX

    def grad(self, X):
        row_diff = X.sum(axis=1) - self.a
        col_diff = X.sum(axis=0) - self.b
        grad = np.concatenate([row_diff, col_diff], axis=0)
        return np.linalg.norm(grad)

    def optimize(self, tol, rho=None, first_stage_tol=1e-3, verbose=True):
        # initialization
        X = np.ones((self.m, self.n))
        row_sum = np.maximum(self.K.sum(axis=1, keepdims=True), 1e-300)
        g = np.zeros(self.n)
        f = np.zeros(self.m)
        P = self.K / row_sum
        t_total = 0.0
        X1 = P * self.a[:, None]
        gnorm = self.grad(X1)
        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X, newton_iter=0, cg_time=0, first_stage_iter=0)
        # _ = self.kkt_err(X, g)

        fixed_term = self.K
        log_fixed_term = self.logK

        # first-order method
        t_start = timeit.default_timer()
        if self.skip_first_stage:
            iter1 = 0
            X = P * self.a[:, None]
        else:
            warmstart = [f, g / self.eta]
            P, X, iter1, u, v = self.first_phase(self.C, warmstart=warmstart, stopThr=first_stage_tol)
            g = self.eta * np.log(v)
            g -= np.mean(g)


        t_end = timeit.default_timer()
        t_total += t_end - t_start

        gnorm = self.grad(X)
        gradient = X.sum(axis=0) - self.b

        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X, newton_iter=0, cg_time=0, first_stage_iter=iter1)

        inner_max_iter = 500
        solved = False
        for i in range(inner_max_iter):
            t_start = timeit.default_timer()

            gnorm = np.linalg.norm(gradient)

            if gnorm < tol:
                solved = True
                break

            rho = self.eta / (self.m * self.n) * gnorm
            _, P_sparse = self.Sparsity_introduce(P, rho)

            X_sparse = P_sparse * self.a[:, None]
            column_sum = X_sparse.sum(axis=0)
            X_sparse = sp.csr_matrix(X_sparse)
            P_sparse = sp.csr_matrix(P_sparse)

            H_sv = self.sparsify_hessian(P_sparse, X_sparse, column_sum)
            cg_start_time = timeit.default_timer()
            d, cg_iter = self._solve_direction(H_sv, gradient, gnorm)
            cg_end_time = timeit.default_timer()

            alpha, dual_obj = self.line_search_Armijo_(g, log_fixed_term, fixed_term, gradient, d)
            g += alpha * d
            P, X = self.update_P_X(g, fixed_term)

            # gradient used to calculate hessian
            column_sum = X.sum(axis=0)
            gradient = column_sum - self.b

            t_end = timeit.default_timer()
            t_total += t_end - t_start

            cg_time = cg_end_time - cg_start_time
            self.record(t_total=t_total, cg_iter=cg_iter, gnorm=np.linalg.norm(gradient), X=X, cg_time=cg_time, newton_iter=i, first_stage_iter=iter1)

            if verbose and i % 1 == 0:
                print(f"iter = {i:5d}, grad_norm = {gnorm:2.4e}, "
                      f"col_resid = {self.history['col_resid'][-1]:2.2e}, "
                      f"time = {t_total:3.2f}")

        if not solved:
            warnings.warn("The algorithm does not reach the required tolerance, you may increase the regularization parameter.")

        return X

    def Sparsity_introduce(self, P, rho):
        mask = P > rho
        mask[self.argmax, :] = True
        P_sparse = P * mask
        row_sum = P_sparse.sum(axis=1, keepdims=True)
        if np.any(row_sum == 0.0):
            warnings.warn("Sparsity P: A row in P has all zero entries. It is better to increase regularization parameter eta.")
            zero_indices = np.where(row_sum == 0)[0]
            P[zero_indices, :] = np.random.uniform(low=0.0, high=1.0, size=(len(zero_indices), self.n))
            row_sum = P.sum(axis=1, keepdims=True)
        P_sparse = P_sparse / row_sum
        return mask, P_sparse

    def dual_obj_cal_normalize(self, g, fixed_term):
        max_g = np.max(g)
        e = np.exp((g - max_g) / self.eta)  # (n,)
        numerator = fixed_term * e  # (m,n)
        row_sum = np.maximum(numerator.sum(axis=1), 1e-300)  # (m,)
        # obj = -g^T b + η Σ_i a_i log(row_sum_i) + max_g Σ_i a_i

        return -np.dot(g, self.b) + self.eta * np.sum(self.a * np.log(row_sum)) + max_g

    def line_search_Armijo_(self, g, log_fixed_term, fixed_term, cur_grad_g, direction, sigma=0.8, gamma=1e-4):
        alpha = 1.0
        iter_num = 0
        cur_obj = self.dual_obj_cal_normalize(g, fixed_term)
        # print("current objective value is ", cur_obj)
        inner_prod = np.dot(cur_grad_g, direction)
        new_dual_obj = 0.0
        # print("inner product is ", inner_prod)
        if inner_prod > 0:
            print("Inner product is positive with value ", inner_prod)
            assert inner_prod < 0, "Inner product should be negative"
        # cur_grad_g = self.b - np.sum(self.X, axis=0, keepdims=False)
        while iter_num <= 100:
            new_g = g + alpha * direction
            new_dual_obj = self.dual_obj_cal_normalize(new_g, fixed_term)
            # print("new objective value is ", new_dual_obj)
            if new_dual_obj - cur_obj > gamma * alpha * inner_prod:
                alpha *= sigma
            else:
                break
            iter_num += 1
        return alpha, new_dual_obj

    def sparsify_hessian(self, P_sparse, X_sparse, column_sum):

        def H_sv(v):
            # term1 = diag(column_sum) @ v
            term1 = column_sum * v  # (n,)

            # Pv = P @ v
            Pv = P_sparse @ v  # (m,)

            # XtPv = X^T @ (P v)
            XtPv = X_sparse.T @ Pv

            Hv = (term1 - XtPv) / self.eta
            return Hv
        return H_sv

    def _solve_direction(self, H_sv, g, lam):
        n_tot = g.size
        # use LinearOperator + CG
        assert H_sv is not None

        def shift_mv(v):
            return H_sv(v) + lam * v

        # Record iteration count
        it_counter = {"n": 0}

        def cb(xk):
            it_counter["n"] += 1

        Aop = LinearOperator((n_tot, n_tot), matvec=shift_mv, dtype=np.float64)
        p, info = cg(Aop, -g, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        cg_iter = it_counter["n"]
        return p, cg_iter

    def first_phase(self, C_iter, warmstart, stopThr=1e-3):
        X ,log = ot.sinkhorn(
            self.a, self.b, C_iter, self.eta, method=self.first_phase_method,
            numItermax=1000, stopThr=stopThr, log=True, warmstart=warmstart)
        return X / self.a[:, None], X, log['niter'], log['u'], log['v']

class PINSsolver_EOT(optimal_transport):
    def __init__(self, *args, first_phase_method="Sinkhorn", **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol = 1e-11
        self.cg_maxit = max(self.m, self.n) * 100

        self.first_phase_method = first_phase_method

        self.alpha_list = []
        self.loss_list = []

    def internal_x(self, f, g, c_iter):
        value = (np.expand_dims(f, axis=1) + np.expand_dims(g, axis=0) - c_iter) / self.eta
        return value

    def first_phase(self, C_iter, warmstart=None, tol=1e-3):
        X ,log = ot.sinkhorn(
            self.a, self.b, C_iter, self.eta, method=self.first_phase_method,
            numItermax=1000, stopThr=tol, warmstart=warmstart, log=True)
        return X, log['niter'], log['u'], log['v']

    def sparsify(self, X, rho):
        x = np.reshape(X, (X.size,))
        idx = np.argsort(x)
        num_zeros = ceil(x.size * (1 - rho))
        x[idx[0:num_zeros]] = 0.0
        X_sparse = np.reshape(x, X.shape)
        X_sparse = sp.csr_matrix(X_sparse)
        return X_sparse

    def sparsify_hessian(self, row_sum, column_sum, X, rho):
        X_sparse = self.sparsify(X, rho)
        scale = 1.0 / self.eta
        # Construct Hessian
        D_alpha = row_sum * scale                      # diag block (n,)
        D_beta  = column_sum * scale                      # diag block (m-1,)

        X_block = X_sparse * scale          # (n, m-1)
        def Hdelta_mv(v):
            # v = [v_a (n,); v_b (m-1,)]
            va = v[:self.m]
            vb = v[self.m:]
            out_a = D_alpha * va + X_block @ vb                  # (n,)
            out_b = X_block.T @ va + D_beta * vb                 # (m-1,)
            return np.concatenate([out_a, out_b], axis=0)

        return Hdelta_mv

    def _solve_direction(self, Hdelta_mv, g, lam):
        n_tot = g.size
        # use LinearOperator + CG
        assert Hdelta_mv is not None

        def shift_mv(v):
            return Hdelta_mv(v) + lam * v

        # Record iteration count
        it_counter = {"n": 0}

        def cb(xk):
            it_counter["n"] += 1

        Aop = LinearOperator((n_tot, n_tot), matvec=shift_mv, dtype=np.float64)
        p, info = cg(Aop, -g, atol=self.cg_tol, maxiter=self.cg_maxit, callback=cb)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        cg_iter = it_counter["n"]
        return p, cg_iter

    def grad(self, X):
        row_diff = X.sum(axis=1) - self.a
        col_diff = X.sum(axis=0) - self.b
        grad = np.concatenate([row_diff, col_diff], axis=0)
        return np.linalg.norm(grad)

    def optimize(self, tol, rho=1e-1, verbose=True):
        # initializaion
        X = np.ones((self.m, self.n))
        f = np.zeros(self.m)
        g = np.zeros(self.n)
        X1 = np.exp(self.internal_x(f, g, self.C))
        gnorm = self.grad(X1)

        t_total = 0.0
        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X, newton_iter=0, cg_time=0, first_stage_iter=0)
        # kkt_err = self.kkt_err(X, g, f)

        t_start = timeit.default_timer()
        # first phase
        # warmstart = [f, g / self.eta]
        X, iter1, u, v = self.first_phase(self.C, tol=1e-3)
        f, g = self.eta * np.log(u), self.eta * np.log(v)
        t_end = timeit.default_timer()
        t_total += t_end - t_start

        gnorm = self.grad(X)
        self.record(t_total=t_total, cg_iter=0, gnorm=gnorm, X=X, newton_iter=0, cg_time=0, first_stage_iter=iter1)

        # second phase
        # X, f, g, iter2, itercg, timecg = self.sparse_newton(f, g, self.C, maxit=1000, tol=tol, rho=rho)
        Xen = np.sum(X, axis=1)
        Xtem = np.sum(X, axis=0)
        grad_f = Xen - self.a
        grad_g = Xtem - self.b
        grad = np.concatenate([grad_f, grad_g])

        for i in range(100):
            t_start = timeit.default_timer()

            Hdelta_mv = self.sparsify_hessian(Xen, Xtem, X, rho)
            cg_start_time = timeit.default_timer()
            d_fg, cg_iter = self._solve_direction(Hdelta_mv, grad, np.linalg.norm(grad))
            cg_end_time = timeit.default_timer()
            # cg_time.append(cg_end_time - cg_start_time)

            # cg_iter.append(iter)
            d_f = d_fg[0:self.m]
            d_g = d_fg[self.m: self.m + self.n]

            if i == 0:
                internalx = self.internal_x(f, g, self.C)
            else:
                internalx = x

            lya_current = self.lyapunov(f, g, internalx)
            alpha = 1.0
            beta = 0.5
            tol_ls = 1e-4
            inprod = (grad_f * d_f).sum().item() + (grad_g * d_g).sum().item()
            if inprod > 0:
                print("Inner product is positive with value ", inprod)
                assert inprod < 0, "Inner product should be negative"
            while True:
                new_f = f + alpha * d_f
                new_g = g + alpha * d_g
                new_internalx = self.internal_x(new_f, new_g, self.C)
                lya_new = self.lyapunov(new_f, new_g, new_internalx)
                if lya_new <= lya_current + tol_ls * alpha * inprod:
                    break
                alpha *= beta

            step_size = alpha
            f += step_size * d_f
            g += step_size * d_g

            x = new_internalx
            X = np.exp(x)
            Xen = np.sum(X, axis=1)
            Xtem = np.sum(X, axis=0)
            grad_f = Xen - self.a
            grad_g = Xtem - self.b

            grad = np.concatenate([grad_f, grad_g])
            gnorm = np.linalg.norm(grad)

            t_end = timeit.default_timer()
            t_total += t_end - t_start
            cg_time = cg_end_time - cg_start_time

            self.record(t_total=t_total, cg_iter=cg_iter, gnorm=gnorm, X=X, newton_iter=i, cg_time=cg_time, first_stage_iter=iter1)

            if verbose and i % 1 == 0:
                print(f"iter = {i:5d}, grad_norm = {gnorm:2.4e}, "
                      f"row_resid = {self.history['row_resid'][-1]:2.2e}, "
                      f"col_resid = {self.history['col_resid'][-1]:2.2e}, "
                      f"time = {t_total:3.2f}")

            if gnorm < tol:
                break

        return X

    def lyapunov(self, f, g, internalx):
        term1 = - self.eta * (np.exp(internalx)).sum().item()
        term2 = (self.a * f).sum().item() + (self.b * g).sum().item()
        return - term1 - term2

if __name__ == "__main__":
    from Alg.GurobiSolver import *
    import os

    # m, n = 500, 500
    # eta = 1e-4
    # opt = None
    # # np.random.seed(7)
    # a = np.random.rand(m)
    # a = a / np.sum(a)  # [0.54100398 0.01455541 0.44444061]
    # # print("a is ", a)
    # b = np.random.rand(n)
    # b = b / np.sum(b)  # [0.44833981 0.29847674 0.13459504 0.11858842]
    # # print("b is ", b)
    # C = np.random.rand(m, n)
    # C /= np.max(C)

    # guro = Gurobisolver(C, a, b)
    # gt = guro.Gurobi_Solver_original()
    # opt = np.sum(C * gt)
    # print(gt)



    m = n = 1000
    cost_matrix_norm = "Uniform"  # "Square", "Uniform", "Absolute"
    seed = 41
    RESULT_ROOT = f"../Results/synthetic/alg_compare_m{m}_n{n}_rng{seed}"
    RESULT_ROOT = os.path.join(RESULT_ROOT, cost_matrix_norm)
    opt_cache_path = os.path.join(RESULT_ROOT, "opt_ab.npz")
    cache = np.load(opt_cache_path, allow_pickle=False)
    a = cache["a"]
    b = cache["b"]
    opt = float(cache["opt"])

    # ===== cost matrix =====
    if cost_matrix_norm == "Square":
        J = np.arange(n)
        C = (J[None, :] - J[:, None]) ** 2
        C = C / np.max(C)
    elif cost_matrix_norm == "Uniform":
        np.random.seed(seed)
        C = np.random.uniform(0, 1, size=(m, n))
        C = C / np.max(C)

    solver = IBSNsolver_EOT(C=C, eta=1e-4, a=a, b=b, obj_truth=opt, skip_first_stage=False)
    start_time = timeit.default_timer()
    X = solver.optimize(tol=1e-11, first_stage_tol=1e-3)
    # print(X)
    end_time = timeit.default_timer()
    print("time is ", end_time - start_time)
    # X = solver._round_to_marginals(X, a, b)

