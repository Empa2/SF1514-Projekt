import numpy as np

F = lambda x, y: np.array([
    x ** 2 - y + 1,
    2 * x ** 2 + y**2 - 8 
])

J = lambda x, y: np.array([
    [2*x, -1],
    [4*x, 2*y]
])

X = np.array([1, 2], dtype=float)

tol = 1e-10
diff = np.ones(X.shape)
it = 0
max_iter = 100

while np.linalg.norm(diff) > tol and it < max_iter:
    diff = np.linalg.solve(J(*X), -F(*X))
    X += diff
    it += 1
    print(it, X, np.linalg.norm(diff))

if it == max_iter:
    print("Max antal iterationer")


    