import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt
import time

N = 256
length = 125

x = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
y = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
X, Y = np.meshgrid(x, y)

#U = np.exp(-((X-0)**2 + (Y-0)**2))# + np.exp(-((X+20)**2 + (Y+0)**2))
#U = np.exp(U) - 1
#U = np.cos(np.sqrt(X*X + Y*Y))
U = np.ones(np.shape(X))

V = fft2(U)
alpha = 1
beta = 2
sigma = 3
omega = 0.8 * np.exp(-(X*X+Y*Y)/(2*sigma*sigma)) + 0.5
mu = 0.6
chi = 1.7*np.pi

# between 0.2 and 0.25 at N=256, localized target patters
# between 0.1 and 0.15 at N=128, localized target patterns (very small)
delta = 0.13
#g = fft2(np.exp(-(X*X+Y*Y)*(delta*delta)))

h = 0.1

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
bk = np.array(bk)
kx, ky = np.meshgrid(np.array(bk)*2*np.pi/length, np.array(bk)*2*np.pi/length)


g2 = np.exp(-(kx*kx + ky*ky)/(delta*delta)) 
L = -(kx*kx + ky*ky) * (1 + 1j*beta) - mu * np.exp(1j * chi) * g2

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

tmax = 200
nmax = np.round(tmax/h)
nplt = np.floor((tmax/200)/h)


for n in range(1, int(nmax) + 1):
    np.mean(ifft2(V))
    Nv = fft2((1 - 1j*omega) * ifft2(V) - (1 + 1j*alpha)*ifft2(V)*abs(ifft2(V))**2)# + (mu * np.exp(1j*chi) * V) * g# - V
    a = E2*V + Q*Nv
    Na = fft2((1 - 1j*omega) * ifft2(a) - (1 + 1j*alpha)*ifft2(a)*abs(ifft2(a))**2)# + (mu * np.exp(1j*chi) * a) * g
    b = E2*V + Q*Na
    Nb = fft2((1 - 1j*omega) * ifft2(b) - (1 + 1j*alpha)*ifft2(b)*abs(ifft2(b))**2)# + (mu * np.exp(1j*chi) * b) * g
    c = E2*a + Q*(2*Nb-Nv)
    Nc = fft2((1 - 1j*omega) * ifft2(c) - (1 + 1j*alpha)*ifft2(c)*abs(ifft2(c))**2)# + (mu * np.exp(1j*chi) * c) * g
    V = E*V + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3

    if n % nplt == 0:
        t = n*h
        tt.append(t)
        U = ifft2(V).real
        hor.append(U[int(len(U)/2 - 10)])

    count += 1
    print(count)

fig, ax = plt.subplots()
ax.axis('off')
ax.pcolormesh(X, Y, U)

nX, nT = np.meshgrid(tt, x)
hor = np.array(hor)
hor = hor.T
print(np.shape(nX))
print(np.shape(nT))
print(np.shape(hor))

fig2, ax2 = plt.subplots()
ax2.pcolormesh(nX, nT, hor)

plt.show()


