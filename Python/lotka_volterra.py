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
t, y = euler(fun2, np.array([0, 100]), np.array([1, 1]))
y = np.transpose(y)

fig2, ax2 = plt.subplots()
ax2.plot(t, y[0])
ax2.plot(t, y[1])

fig3, ax3 = plt.subplots()
ax3.plot(y[0], y[1])

fig4, ax4 = plt.subplots()
Xdot = lambda t, X: np.array([np.cos(X[0])+X[1]*X[1]*X[0] - X[2],np.sin(X[2]) - X[1]*X[1]*X[0],X[2] + X[1]*X[0] + X[1]])
t, y = euler(Xdot, np.array([0, 10]), np.array([0,1,0]))
y = np.transpose(y)
ax4.plot(y[0], y[2])
plt.show()


