import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.fft import dctn, idctn
from matplotlib.animation import FuncAnimation
import time


a = 2
b = 0.2
mu = 0.001
sigma = (4/3)*(1/np.sqrt(a))

b1 = -0.5
b2 = 0.5
c1 = 0
c2 = 1.5
d2 = 2.75

aux = (4/3)*(1/np.sqrt(a*sigma))

dt = 0.01


N = 256
length = 125#50


bk = [i for i in range(0, N)]
bk = np.array(bk) * 2*np.pi/length
kx, ky = np.meshgrid(bk, bk)
L = -(kx*kx + ky*ky)

delta = 0.05
k = -1 + np.exp(-(kx*kx+ky*ky)*(delta*delta))


x = np.linspace(-length/2, length/2, num=N)
y = np.linspace(-length/2, length/2, num=N)
X, Y = np.meshgrid(x, y)

G = np.exp(-(X**2 + Y**2)/25)
g = np.exp(-(X**2 + Y**2))

I = np.ones(X.shape) * (-b/a)
J = np.ones(X.shape) * (-b/a + (b/a)**2)


fig, ax = plt.subplots()
ax.axis('off')
plot = ax.imshow(I)

def init():
    return plot

t = time.time()
count = 0

def animate(i):    
    global I, J, t, count

    t0 = time.time()
    count += 1
    print(t - t0, '\t', count)
    t = t0

    dscI = dctn(I)
    dscJ = dctn(J)

    nf1 = ((I - I**3) - J) / sigma + c1*G*J
    nf2 = a*I + b + c2*G*I

    I = idctn((dscI + dt * (b1*k*dscJ + dctn(nf1))) / (1 - dt*L))
    J = idctn((dscJ + dt * (d2*L*dscI + dctn(nf2) + b2*k*dscI)) / (1 - dt*L))

    plot = ax.imshow(I.real)
    return plot

anim = FuncAnimation(fig, animate, init_func=init, blit=False, save_count=100, cache_frame_data=False)
anim.save('fhn_imp_exp.gif')
#plt.show()


