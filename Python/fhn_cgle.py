import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.fft import dctn, idctn
from matplotlib.widgets import Slider, Button


a = 2
b = 0.2
mu = 0
sigma = (4/3)*(1/np.sqrt(a))

b1 = -0.5
b2 = 0.5
c1 = 0
c2 = 1.5
d2 = 2.75

aux = (4/3)*(1/np.sqrt(a*sigma))

dt = 0.01


N = 256
length = 50


bk = [i for i in range(0, N)]
#bk = [i for i in range(0, int(N/2))]
#bk[len(bk):] = [0]
#bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
bk = np.array(bk) * 2*np.pi/length
kx, ky = np.meshgrid(bk, bk)
L = -(kx*kx + ky*ky)

delta = 0.05
k = -1 + np.exp(-(kx*kx+ky*ky)*(delta*delta))


x = np.linspace(-length/2, length/2, num=N)
y = np.linspace(-length/2, length/2, num=N)
X, Y = np.meshgrid(x, y)

G = np.exp(-(X**2 + Y**2))
g = np.exp(-(X**2 + Y**2))

I = np.ones(X.shape) * (-b/a) - 0.01
J = np.ones(X.shape) * (-b/a + (b/a)**2)


t = 0
tt = []
hor = []
count = 0

tmax = 50
nmax = np.round(tmax/dt)
nplt = np.floor((tmax/400)/dt)

centerI = []
centerJ = []

u = I
v = J

for n in range(1, int(nmax) + 1): #int(nmax) + 1
    
    dscI = dctn(I)
    dscJ = dctn(J)

    nf1 = ((I - I**3) - J) / sigma + c1*G*J
    nf2 = a*I + b + c2*G*I

    I = idctn((dscI + dt * (b1*k*dscJ + dctn(nf1))) / (1 - dt*L))
    J = idctn((dscJ + dt * (d2*L*dscI + dctn(nf2) + b2*k*dscI)) / (1 - dt*L))

    count += 1
    #print(count, '\t', np.sum(I))
    print(count)

    centerI.append(abs(I[int(len(I)/2)][int(len(I)/2)]))
    centerJ.append(abs(J[int(len(J)/2) + 1][int(len(J)/2)]))

    if n % nplt == 0:
        t = n*dt
        tt.append(t)
        hor.append(J[int(len(I)/2)].real)

#print((1 + d1*(h*(L**2))))

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, I.real)

nT, nX = np.meshgrid(tt, x)
hor = np.array(hor).T
print(np.shape(nX))
print(np.shape(nT))
print(np.shape(hor))

fig2, ax2 = plt.subplots()
ax2.pcolormesh(nT, nX, hor)

fig3, ax3 = plt.subplots()
#print(centerI)
#print(centerJ)
ax3.plot(centerI, centerJ)


# draw
plt.show()


