import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi

np.set_printoptions(linewidth=np.inf)

cd = 0.3
mass = 0.007
air_density = 1.293
area = 4.5e-5
drag_factor = cd * air_density * area / (2*mass)
g = np.array([0, -9.8])

N = 10
v0 = 350 
target = np.array([1200, 20])


def getA(t):
    T = np.stack([t**i for i in range(1, t.size+1)]).T
    dT = np.stack([i*t**(i-1) for i in range(1, t.size+1)]).T
    return dT @ np.linalg.inv(T)


def fun(y):
    v = np.sqrt(y[:, 1] ** 2 + y[:, 3]**2)
    ax = -drag_factor*v*y[:, 1] + g[0]
    ay = -drag_factor*v*y[:, 3] + g[1]
    return np.stack([y[:, 1], ax, y[:, 3], ay], axis=1)


def fun_jac(y):
    v = np.sqrt(y[:, 1] ** 2 + y[:, 3]**2)
    dy = np.zeros(y.shape * 2)

    indices = range(y.shape[0])
    dy[indices, 0, indices, 1] = 1
    dy[indices, 2, indices, 3] = 1
    dy[indices, 1, indices, 1] = -drag_factor*( v + y[:, 1]**2/v ) 
    dy[indices, 3, indices, 3] = -drag_factor*( v + y[:, 3]**2/v ) 
    dy[indices, 1, indices, 3] = -drag_factor*y[:, 1]*y[:, 3]/v
    dy[indices, 3, indices, 1] = -drag_factor*y[:, 3]*y[:, 1]/v

    return dy


def residual(y, theta, ts):
    y0 = np.array([0, np.cos(theta)*v0, 0, np.sin(theta)*v0])
    return fun(y) - A @ (y - y0) * ts


def residual_jac(y, ts):
    return fun_jac(y) - A[:, None, :, None] * np.eye(4, 4)[None, :, None, :] * ts

theta = np.arctan(target[1] / target[0])
time = np.sum(target**2)**0.5 / v0
ts_inv = time / (N-1)

t, weights = roots_jacobi(n=N, alpha=1, beta=1)
t = (np.append(t, 1) + 1) / 2
y = np.zeros((t.size, 4))
y[:, 0] = t*target[0]
y[:, 2] = t*target[1]
y[:, 1] = np.cos(theta) * v0
y[:, 3] = np.sin(theta) * v0

A = getA(t)

for i in range(1000):
    y0 = np.array([0, np.cos(theta)*v0, 0, np.sin(theta)*v0])
    y[-1, 0] = target[0]
    y[-1, 2] = target[1]

    res = residual(y, theta, ts_inv)
    jac = residual_jac(y, ts_inv)

    print("čas zásahu", f"{1/ts_inv:.6f}", "s   elevační úhel:", f"{theta:.6f}", "Rad   error:", np.sum(res ** 2))

    ax = plt.gca()
    ax.set_aspect(4, adjustable='box')
    plt.ylim(-40, 150) 
    plt.plot(np.concatenate(([0], y[:, 0])), np.concatenate(([0], y[:, 2])))
    plt.show()

    jac[:, :, -1, 0] = -A @ (y - y0)
    jac[:, :, -1, 2] = 0
    jac[:, 1, -1, 2] = -np.sum(A, axis=1)*ts_inv*np.sin(theta)*v0
    jac[:, 3, -1, 2] = +np.sum(A, axis=1)*ts_inv*np.cos(theta)*v0

    dy = np.tensordot(np.linalg.tensorinv(jac), res)
    theta -= dy[-1, 2]
    ts_inv -= dy[-1, 0]
    y -= dy
