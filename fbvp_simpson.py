
import numpy as np
import termplotlib as tpl
import math

cd = 0.3
mass = 0.007
air_density = 1.293
area = 4.561e-5
drag_factor = cd * air_density * area / (2*mass)
g = np.array([0, -9.8])


def fun(t,  y):
    v = np.sqrt(y[:, 2] ** 2 + y[:, 3]**2)
    ax = -drag_factor*v*y[:, 2] + g[0]
    ay = -drag_factor*v*y[:, 3] + g[1]
    return np.stack([y[:, 2], y[:, 3], ax, ay], axis=1)


def fun_jac(t, y):
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


def residual(t, y, ts):
    f = fun(ts*t, y)
    yc = 1/2*(y[:-1] + y[1:]) + ts/8*(f[:-1] - f[1:])
    z = f[:-1] + f[1:] + 4*fun(ts*t, yc)
    return  y[1:] - y[:-1] - ts/6*z, yc, z, f


def residual_jac(t, y, ts, yc):
    f_jac = fun_jac(t*ts, y)
    fc_jac = fun_jac(t*ts, yc) #t wrong

    m,n = y.shape[0], y.shape[1]
    id = np.eye(n)[None,:,None,:] * np.eye(m-1)[:, None, :, None]

    dres = np.zeros((m-1, n, m, n))
    dres[:,:,1:,:] += id - ts/6*(f_jac[1:,:,1:,:])

    np.set_printoptions(linewidth=np.inf)

    #print(fc_jac)
    print()
    print(fc_jac.reshape(8, 8))
    print()
    print(f_jac.reshape(12, 12))
    print()
    print(dres.reshape(8,12))

    dres[:,:,1:,:] -= ts/3*np.tensordot(fc_jac, id - ts/4*f_jac[1:,:,1:,:])

    dres[:,:,:-1,:] -= id + ts/6*f_jac[:-1,:,:-1,:]
    dres[:,:,:-1,:] -= ts/3*np.tensordot(fc_jac, (id + ts/4*f_jac[:-1,:,:-1,:]))

    return dres, fc_jac


N = 3 
v0 = 350 
theta = 0.1
target = np.array([1200, 20])
time = 12
ts = time / N

t = np.linspace(0, 1, N)
y = np.zeros((N, 4))
y[:, 0:2] = np.linspace(0, target, N)
y[:, 2] = v0 * math.cos(theta)
y[:, 3] = v0 * math.sin(theta)


for i in range(100):
    y[-1, 0:2] = target
    y[0, :] = np.array([0, 0, math.cos(theta) * v0, math.sin(theta) * v0 ])

    res, yc, z, f = residual(t, y, ts)
    jac, fc_jac = residual_jac(t, y, ts, yc)
    jac[:,:, -1, 0] = -z/6 -ts/12*np.tensordot(fc_jac, (f[:-1] - f[1:])) # dres/dts
    jac[:,:, -1, 1] = -jac[:,:,0,2]*math.sin(theta)*v0 + jac[:,:,0,3]*math.cos(theta)*v0 # dres/dtheta

    fig = tpl.figure()
    fig.plot(y[:, 0], y[:, 1], width=100, height=20)
    fig.show()

    print(ts, theta, np.sum(res ** 2))

    dy = np.tensordot(np.linalg.tensorinv(jac[:, :, 1:, :]), res)
    y[1:,:] -= dy
    ts -= dy[-1, 0] 
    theta -= dy[-1,1]



    #N theta
    #3  0.2810180934614415
    #4  0.2518119632647842
    #5  0.24878970404487957
    #6  0.2481361414570262  0.2478370731441649 
    #7  0.2479417221940892
    #8  0.24787068982116278
    #9  0.24784060569304778 0.2478076305150177
    #10 0.24782635193968916
    #11 0.24781897628143842
    #12 0.24781487892161735
    #15 0.24781001424541632


    #for i,j in np.ndindex(y.shape):
    #    dt = 1e-5
    #    y[i,j] += dt
    #    res2,_,_,_,_ = residual(t, y, ts)
    #    y[i,j] -= 2*dt
    #    res1,_,_,_,_ = residual(t, y, ts)
    #    y[i,j] += dt
    #    
    #    for k,l in np.ndindex(yc.shape):
    #        dd = (res2[k,l] - res1[k,l]) / (2*dt)
    #        assert(1e-3 > abs(jac[k,l,i,j] - dd ))

    #jac[:,:, -1, 1] = -jac[:,:,0,1]*math.sin(theta)*v0 + jac[:,:,0,3]*math.cos(theta)*v0
    #jac[:,:, -1, 0] = -z/6 -ts/12*np.tensordot(fc_jac, (f[:-1] - f[1:]))

    #dt = 1e-5
    #setbc(y, theta+dt, target)
    #res2, _, _, _, _ = residual(t, y, ts)
    #setbc(y, theta-dt, target)
    #res1, _, _, _, _ = residual(t, y, ts)

    #for k,l in np.ndindex(yc.shape):
    #    dd = (res2[k,l] - res1[k,l]) / (2*dt)
    #    assert(1e-3 > abs(jac[k,l,-1,1] - dd) )

    #dt = 1e-5
    #res2, _, _, _, _ = residual(t, y, ts+dt)
    #res1, _, _, _, _ = residual(t, y, ts-dt)

    #for k,l in np.ndindex(yc.shape):
    #    dd = (res2[k,l] - res1[k,l]) / (2*dt)
    #    assert(2e-3 > abs(jac[k,l,-1,0] - dd) )
