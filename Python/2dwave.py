import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# first create a wave equation
def wave(t, x, y):
    w = 1.0
    k = 1.0
    return np.cos(w*np.sqrt(x*x+y*y) - k*0.1*t)


# create a meshgrid
x = np.linspace(-20,20, num = 100)
y = np.linspace(-20,20, num = 100)
xx, yy = np.meshgrid(x, y)
zz = wave(4,xx,yy)

# create plot
fig, ax = plt.subplots()

plot = ax.imshow(zz)
print(plot)
def init():
    return plot

def update(t):
    zz = wave(t,xx,yy)
    print(t)
    plot = ax.imshow(zz)
    return plot

anim = FuncAnimation(fig, update, init_func=init, frames=100, interval=20, blit=False)
plt.show()

