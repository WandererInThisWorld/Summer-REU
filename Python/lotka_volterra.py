import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

cos = np.cos

def LotkaVolterraFun(V, a, b, c, d):
    X = V[0]
    Y = V[1]
    return np.array([a*X - b*X*Y, c*X*Y - d*Y])

def LotkaVolterraFun2(t, V, a, b, c, d):
    X = V[0]
    Y = V[1]
    return np.array([a*X - b*X*Y, c*X*Y - d*Y])


def euler(f, t, y):
    dt = 0.001
    time = np.linspace(t[0], t[1], num=int((t[1] - t[0])/dt)) # change number

    p_t = [y]
    for idx in range(1, len(time)):
        p_t.append(p_t[-1] + dt*f(time[idx],p_t[-1]))

    return time, p_t


def runge_kutta(f, t, y):
    h = 0.05
    time = np.linspace(t[0], t[1], int((t[1] - t[0])/h))

    path = [y]
    for idx in range(1, len(time)):
        k1 = f(time[idx], path[-1])
        k2 = f(time[idx] + h/2, path[-1] + k1*h/2)
        k3 = f(time[idx] + h/2, path[-1] + k2*h/2)
        k4 = f(time[idx] + h, path[-1] + k3*h)

        m = k1/6 + k2/3 + k3/3 + k4/6

        path.append(path[-1] + m*h)

    return time, path        

def mod_runge_kutta(f, t, y):
    path = []

a = 2
b = 3
c = 1
d = 1


fun = lambda V : LotkaVolterraFun(V, a, b, c, d)
Veq = fsolve(fun, np.array([1, 5]))
print(Veq)

x = np.linspace(0, 5, num=30)
y = np.linspace(0, 5, num=30)
X, Y = np.meshgrid(x, y)
U = a*X - b*X*Y
V = c*X*Y - d*Y

fig1, ax1 = plt.subplots()
ax1.quiver(X, Y, U/np.sqrt(U*U + V*V), V/np.sqrt(U*U + V*V))
ax1.scatter(Veq[0], Veq[1])
ax1.scatter(0, 0)

fun2 = lambda t, V : LotkaVolterraFun2(t, V, a, b, c, d)
t, y = runge_kutta(fun2, np.array([0, 50]), np.array([1, 1]))
y = np.transpose(y)

fig2, ax2 = plt.subplots()
ax2.plot(t, y[0])
ax2.plot(t, y[1])

fig3, ax3 = plt.subplots()
ax3.plot(y[0], y[1])

fig4, ax4 = plt.subplots()
Xdot = lambda t, X: np.array(-2*X[0], X[1])
t, y = euler(Xdot, np.array([0, 2]), np.array([3,0]))
y_eul = np.transpose(y)
ax4.plot(t, y_eul[0])
t, y = runge_kutta(Xdot, np.array([0, 2]), np.array([3, 0]))
y_rk = np.transpose(y)
ax4.plot(t, y_rk[0])
plt.show()
