from Alg.Algorithm_Import import *
import matplotlib.pyplot as plt
import os

m, n = 500, 500
eta = 1e-1
opt = None
np.random.seed(1)
a = np.random.rand(m)
a = a / np.sum(a)
b = np.random.rand(n)
b = b / np.sum(b)
C = np.random.rand(m, n)

# J = np.arange(n)
# C = (J[None, :] - J[:, None]) ** 2
# C = C / np.max(C)

solver = IBSNsolver(C=C, eta=1e-3, a=a, b=b)
X = solver.optimize(max_iter=300, tol=1e-11)
X = solver._round_to_marginals(X, a, b)

print("row difference is ", np.linalg.norm(X.sum(axis=1) - a))
print("col difference is ", np.linalg.norm(X.sum(axis=0) - b))
print("objective value is ", np.sum(C * X))