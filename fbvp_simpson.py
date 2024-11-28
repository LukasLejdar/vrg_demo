
import numpy as np
import matplotlib.pyplot as plt
import math

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

def fun(y):
    v = np.sqrt(y[:, 2] ** 2 + y[:, 3]**2)
    ax = -drag_factor*v*y[:, 2] + g[0]
    ay = -drag_factor*v*y[:, 3] + g[1]
    return np.stack([y[:, 2], y[:, 3], ax, ay], axis=1)


def fun_jac(y):
    dy = np.zeros(y.shape*2)
    v = np.sqrt(y[:, 2] ** 2 + y[:, 3]**2)
    v[v == 0] = 1

    indices = range(y.shape[0])
    dy[indices, 0, indices, 2] = 1
    dy[indices, 1, indices, 3] = 1
    dy[indices, 2, indices, 2] = -drag_factor*( v + y[:, 2]**2/v ) 
    dy[indices, 3, indices, 3] = -drag_factor*( v + y[:, 3]**2/v ) 
    dy[indices, 2, indices, 3] = -drag_factor*y[:, 2]*y[:, 3]/v
    dy[indices, 3, indices, 2] = -drag_factor*y[:, 3]*y[:, 2]/v

    return dy


def residual(y, ts):
    f = fun(y)
    yc = 1/2*(y[:-1] + y[1:]) + ts/8*(f[:-1] - f[1:])
    z = f[:-1] + f[1:] + 4*fun(yc)
    return  y[1:] - y[:-1] - ts/6*z, yc, z, f


def residual_jac(y, ts, yc):
    f_jac = fun_jac(y)
    fc_jac = fun_jac(yc)

    m,n = y.shape[0], y.shape[1]
    id = np.eye(n)[None,:,None,:] * np.eye(m-1)[:, None, :, None]

    dres = np.zeros((m-1, n, m, n))
    dres[:,:,1:,:] += id - ts/6*(f_jac[1:,:,1:,:])
    dres[:,:,1:,:] -= ts/3*np.tensordot(fc_jac, id - ts/4*f_jac[1:,:,1:,:])

    dres[:,:,:-1,:] -= id + ts/6*f_jac[:-1,:,:-1,:]
    dres[:,:,:-1,:] -= ts/3*np.tensordot(fc_jac, (id + ts/4*f_jac[:-1,:,:-1,:]))

    return dres, fc_jac


theta = np.arctan(target[1] / target[0])
time = np.sum(target**2)**0.5 / v0
ts = time / (N-1)

t = np.linspace(0, 1, N)
y = np.zeros((N, 4))
y[:, 0:2] = target[None, :] * t[:, None]
y[:, 2] = v0 * math.cos(theta)
y[:, 3] = v0 * math.sin(theta)

hit_time = np.roots([0.25*g[1]**2, -g[1]*target[1] - v0**2, target[0]**2 + target[1]**2 ])
print(np.sqrt(hit_time))

for i in range(100):
    y[-1, 0:2] = target
    y[0, :] = np.array([0, 0, math.cos(theta) * v0, math.sin(theta) * v0 ])

    res, yc, z, f = residual(y, ts)
    jac, fc_jac = residual_jac(y, ts, yc)
    jac[:,:, -1, 0] = -z/6 -ts/12*np.tensordot(fc_jac, (f[:-1] - f[1:])) # dres/dts
    jac[:,:, -1, 1] = -jac[:,:,0,2]*math.sin(theta)*v0 + jac[:,:,0,3]*math.cos(theta)*v0 # dres/dtheta

    print("čas zásahu", f"{ts*(N-1):.6f}", "s   elevační úhel:", f"{theta:.6f}", "Rad   error:", np.sum(res ** 2))    

    ax = plt.gca()
    ax.set_aspect(4, adjustable='box')
    plt.ylim(-40, 150) 
    plt.plot(y[:, 0], y[:, 1])
    plt.show()

    dy = np.tensordot(np.linalg.tensorinv(jac[:, :, 1:, :]), res)
    y[1:,:] -= dy
    ts -= dy[-1, 0] 
    theta -= dy[-1,1]
