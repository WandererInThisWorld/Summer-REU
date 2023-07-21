import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.fft import dctn, idctn
from matplotlib.widgets import Slider, Button


a = 1
b = 0.2
sigma = 0.1


b1 = -0.1
b2 = 0.1
c1 = 0
c2 = 0.5
d2 = 0
d3 = 0

'''
d2 = 0
b1 = 0
b2 = 0
c1 = 0
c2 = 0.5
d3 = 0
'''

dt = 0.01


N = 256
length = 120


bk = [i for i in range(0, N)]
#bk = [i for i in range(0, int(N/2))]
#bk[len(bk):] = [0]
#bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
bk = np.array(bk) * 2*np.pi/length
kx, ky = np.meshgrid(bk, bk)
L = -(kx*kx + ky*ky)
delta = 100
k = -1 + np.exp(-(kx*kx+ky*ky)*(delta*delta))


x = np.linspace(-length/2, length/2, num=N)
y = np.linspace(-length/2, length/2, num=N)
X, Y = np.meshgrid(x, y)

g = np.exp(-(X**2 + Y**2))

u = np.ones(X.shape) * (-b/a)
v = np.ones(X.shape) * (-b/a + (b/a)**2)


t = 0
tt = []
hor = []
count = 0

tmax = 60
nmax = np.round(tmax/dt)
nplt = np.floor((tmax/400)/dt)

centerI = []
centerJ = []
frame = u

for n in range(1, int(nmax) + 1):
    
    dscu = dctn(u)
    dscv = dctn(v)

    nf1 = ((u - u**3) - v) / sigma + c1*g*v
    nf2 = a*u + b + c2*g*u

    u = idctn((dscu + dt * (b1*k*dscv + dctn(nf1))) / (1 - dt*L))
    v = idctn((dscv + dt * (d2*L*dscu + dctn(nf2) + b2*k*dscu)) / (1 - d3*dt*L))

    count += 1
    print(count)

    centerI.append(abs(u[int(len(u)/2)][int(len(u)/2)]))
    centerJ.append(abs(v[int(len(v)/2) + 1][int(len(v)/2)]))

    if n % nplt == 0:
        t = n*dt
        tt.append(t)
        hor.append(v[int(len(u)/2)].real)

    if n*dt <= 52.58:
        frame = u


fig, ax = plt.subplots()
ax.axis('off')
ax.pcolormesh(X, Y, frame.real)

nT, nX = np.meshgrid(tt, x)
hor = np.array(hor).T
print(np.shape(nX))
print(np.shape(nT))
print(np.shape(hor))

fig2, ax2 = plt.subplots()
ax2.pcolormesh(nT, nX, hor)

fig3, ax3 = plt.subplots()
ax3.plot(centerI, centerJ)


# draw
plt.show()


