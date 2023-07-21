import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import time

N = 256
length = 125

x = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
y = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
X, Y = np.meshgrid(x, y)

alpha = 4.2164 #0.970983543414647
beta = 0 #1.0013267791463547
omega = -alpha #0.5719329259465489 * np.exp(-(X*X+Y*Y)/50) - 0.5946035575013606
dw = -0.0456*9
Rez = 0
Imz = -0.01
z = 3.8 * (Rez + 1j * Imz)
mu = abs(z)
chi = np.log(z)
g = dw * np.exp(-(X**2 + Y**2))
h = 0.01

U = np.ones(np.shape(X))
V = fft2(U)

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
bk = np.array(bk)
kx, ky = np.meshgrid(np.array(bk)*2*np.pi/length, np.array(bk)*2*np.pi/length)

L = -(kx*kx + ky*ky) * (1 + 1j*beta) + (1 - 1j*omega)
Nonl = - mu * np.exp(1j * chi) * np.exp(-(kx**2 + ky**2)*1000)
L = L + Nonl

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
LR = h*newL + newr
LR = np.array(LR)

print(np.shape(newL))
print(np.shape(newr))
print(np.shape(LR))


tQ = (np.exp(LR/2) - 1)/LR
Q = h*np.mean(tQ, axis=1).real

tf1 = (-4-LR+np.exp(LR)*(4-3*LR+LR**2))/(LR**3)
f1 = h*np.mean(tf1, axis=1).real

tf2 = (2+LR+np.exp(LR)*(-2+LR))/(LR**3)
f2 = h*np.mean(tf2, axis=1).real

tf3 = (-4-3*LR-LR**2+np.exp(LR)*(4-LR))/(LR**3)
f3 = h*np.mean(tf3, axis=1).real

size = len(bk)
Q = np.reshape(Q, (size, size))
f1 = np.reshape(f1, (size, size))
f2 = np.reshape(f2, (size, size))
f3 = np.reshape(f3, (size, size))
L = np.reshape(L, (size, size))

t = 0
tt = []
hor = []
count = 0

tmax = 51
nmax = np.round(tmax/h)
nplt = np.floor((tmax/400)/h)
frame = []

for n in range(1, int(nmax) + 1):
    Nv = fft2(-(1 + 1j*alpha)*ifft2(V)*abs(ifft2(V))**2) - fft2(1j * g * ifft2(V))
    a = E2*V + Q*Nv
    Na = fft2(-(1 + 1j*alpha)*ifft2(a)*abs(ifft2(a))**2) - fft2(1j * g * ifft2(a))
    b = E2*V + Q*Na
    Nb = fft2(-(1 + 1j*alpha)*ifft2(b)*abs(ifft2(b))**2) - fft2(1j * g * ifft2(b))
    c = E2*a + Q*(2*Nb-Nv)
    Nc = fft2(-(1 + 1j*alpha)*ifft2(c)*abs(ifft2(c))**2) - fft2(1j * g * ifft2(c))
    V = E*V + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3

    if n % nplt == 0:
        t = n*h
        tt.append(t)
        U = ifft2(V).real
        hor.append(U[int(len(U)/2)])

    if n*h < 12:
        frame = ifft2(V).real

    count += 1
    print(count)

fig, ax = plt.subplots()
ax.axis('off')
U = ifft2(V).real
ax.pcolormesh(X, Y, frame)

print(mu)
print(chi)

tt = tt

nX, nT = np.meshgrid(tt, x)
hor = np.array(hor)
hor = hor.T
print(np.shape(nX))
print(np.shape(nT))
print(np.shape(hor))

fig2, ax2 = plt.subplots()
ax2.pcolormesh(nX, nT, hor)

plt.show()

