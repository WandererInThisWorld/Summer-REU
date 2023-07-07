# From paper:"Laser induced target patterns in the oscillatory CO Oxidation on Pt(110)"
# This code uses the globally coupled cgl: equations (5) and (6)
# This code reproduces figure 6

import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import time

N = 128
length = 125

x = 125*np.array([i for i in range(-int(N/2), int(N/2))])/(N/2)
y = 125*np.array([i for i in range(-int(N/2), int(N/2))])/(N/2)
X, Y = np.meshgrid(x, y)

U = np.ones(np.shape(X))

V = fft2(U)
alpha = 1
beta = 2
sigma = 3
omega =  0.5 + 0.8 * np.exp(-(X*X+Y*Y)/(2*sigma*sigma))
mu = 0.6
chi = 1.7*np.pi
g = 1 - np.exp(-(X*X+Y*Y)/(2*sigma*sigma))
h = 0.1

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
bk = np.array(bk)*(2*np.pi)/length
kx, ky = np.meshgrid(bk, bk)
#kx = np.array(bk)/16
#ky = np.array(bk)/16


L = -(kx*kx + ky*ky) * (1 + 1j*beta)

E = np.exp(h*L)
E2 = np.exp(h*L/2)

M = 16

print(bk.shape)
print(len(bk))
L = np.reshape(L, len(bk)*len(bk))

r = np.exp(1j * np.pi * ((np.array([i for i in range(1, M+1)]))-0.5)/M)

newL = np.array([L for i in range(0,M)])
newL = np.transpose(newL)
newr = np.array([r for i in range(0, N*N)])
LR = []
'''
for i in range(len(newL)):
    LR.append(h*newL[i] + newr)
'''
LR = h*newL + newr
LR = np.array(LR)

print(np.shape(newL))
print(np.shape(newr))
print(np.shape(LR))


tQ = (np.exp(LR/2) - 1)/LR

Q = []
for row in tQ:
    Q.append(h * np.mean(row).real)
Q = np.array(Q)

tf1 = (-4-LR+np.exp(LR)*(4-3*LR+LR**2))/(LR**3)
f1 = []
for row in tf1:
    f1.append(h * np.mean(row).real)
f1 = np.array(f1)

tf2 = (2+LR+np.exp(LR)*(-2+LR))/(LR**3)
f2 = []
for row in tf2:
    f2.append(h * np.mean(row).real)
f2 = np.array(f2)

tf3 = (-4-3*LR-LR**2+np.exp(LR)*(4-LR))/(LR**3)
f3 = []
for row in tf3:
    f3.append(h * np.mean(row).real)
f3 = np.array(f3)

size = len(bk)
Q = np.reshape(Q, (size, size))
f1 = np.reshape(f1, (size, size))
f2 = np.reshape(f2, (size, size))
f3 = np.reshape(f3, (size, size))
L = np.reshape(L, (size, size))


tmax = 100
nmax = np.round(tmax/h)
nplt = 1#np.floor((tmax/100)/h)

t=[0]
tt=0
u=[U[round(N/2)]]

for n in range(1, int(nmax) + 1):
   
    mean = V[0]
    np.mean(ifft2(V))
    Nv = fft2((1 - 1j*omega) * ifft2(V) - (1 + 1j*alpha)*ifft2(V)*abs(ifft2(V))**2 -mu*np.exp(1j*chi)*V[0][0]/length**2 *g)
    a = E2*V + Q*Nv
    Na = fft2((1 - 1j*omega) * ifft2(a) - (1 + 1j*alpha)*ifft2(a)*abs(ifft2(a))**2 -mu*np.exp(1j*chi)*a[0][0]/length**2 *g)
    b = E2*V + Q*Na
    Nb = fft2((1 - 1j*omega) * ifft2(b) - (1 + 1j*alpha)*ifft2(b)*abs(ifft2(b))**2 -mu*np.exp(1j*chi)*b[0][0]/length**2 *g)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = fft2((1 - 1j*omega) * ifft2(c) - (1 + 1j*alpha)*ifft2(c)*abs(ifft2(c))**2 -mu*np.exp(1j*chi)*c[0][0]/length**2 *g)
    V = E*V + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    U = ifft2(V).real
    u.append(U[round(N/2)])
    tt = tt+n*h
    t.append(tt)

fig, ax = plt.subplots()
ax.axis('off')
ax.pcolormesh(X, Y, U)

XX, T = np.meshgrid(x, t)
fig, ax2 = plt.subplots()
ax2.axis('off')
ax2.pcolormesh(XX,T,u)

plt.show()
