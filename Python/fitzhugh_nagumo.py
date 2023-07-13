import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def f1(x, y, sigma):
    return ((x - x**3) - y) / sigma

def f2(x, y, a, b):
    return a*x + b


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

x = np.linspace(-2, 2, num=2)
y = np.linspace(-2, 2, num=2)
X, Y = np.meshgrid(x, y)
a = 3
b = a/np.sqrt(3) - 0.8
mu = 1
xdot = f1(X,Y, mu)
ydot = f2(X,Y,a,b)

# xdot = xdot * np.sqrt(xdot*xdot + ydot*ydot)
# ydot = ydot * np.sqrt(xdot*xdot + ydot*ydot)

dt = 0.01
paths = []

mat = ""
for row in range(len(X)):
    for col in range(len(X[0])):
        x0 = X[row][col]
        y0 = Y[row][col]
        path_x = [x0]
        path_y = [y0]

        for idx in range(int(100/dt)):
            dx = 0.01 * f1(path_x[-1], path_y[-1], mu)
            dy = 0.01 * f2(path_x[-1], path_y[-1], a, b)
            path_x.append(path_x[-1] + dx)
            path_y.append(path_y[-1] + dy)
        
        paths.append((path_x, path_y))
            
for path in paths:
    path_x, path_y = path
    ax3.plot(path_x, path_y)


#ax1.quiver(X, Y, xdot, ydot)
#ax2.pcolormesh(X, Y, xdot + ydot)

# draw

plt.show()


