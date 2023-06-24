import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib as mlp

N = 128

x = 16*np.pi*np.array([i for i in range(-int(N/2), int(N/2))])/N
y = 16*np.pi*np.array([i for i in range(-int(N/2), int(N/2))])/N
X, Y = np.meshgrid(x, y)

#U = np.exp(-(X*X+Y*Y))
#U = np.exp(U) - 1
#U = np.exp(U) - 1
U = np.cos(np.sqrt(X*X + Y*Y))

V = fft(U)
V1 = fft(U).real
V2 = fft(U).imag
alpha = 1 - 1j
beta = -1 + 0.5j
h = 1/4

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
#kx, ky = np.meshgrid(np.array(bk)/16, np.array(bk)/16)
kx = np.array(bk)/16
ky = np.array(bk)/16


L = -kx*kx - ky*ky
E = np.exp(h*L)
E2 = np.exp(h*L/2)

M = 16
r = np.exp(1j * np.pi * ((np.array([i for i in range(1, M+1)]))-0.5)/M)

newL = np.array([L for i in range(0,M)])
newL = np.transpose(newL)
newr = np.array([r for i in range(0, N)])
LR = h*newL + newr

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


UU = [U]
tt = [0]
tmax = 100
nmax = np.round(tmax/h)
nplt = 1#np.floor((tmax/100)/h)


for n in range(1, int(nmax) + 1):
    t = n*h
    Nv = fft(alpha * ifft(V) + beta*ifft(V)*abs(ifft(V))**2)

    a = E2*V + Q*Nv
    Na = fft(alpha * ifft(a) + beta*ifft(a)*abs(ifft(a))**2)
    b = E2*V + Q*Na
    Nb = fft(alpha * ifft(b) + beta*ifft(b)*abs(ifft(b))**2)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = fft(alpha * ifft(c) + beta*ifft(c)*abs(ifft(c))**2)
    V = E*V + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    U = ifft(V).real

    UU.append(U)
    tt.append(t)

fig1, ax1 = plt.subplots()
ax1.pcolormesh(X, Y, U)

'''
fig2, ax2 = plt.subplots()
ax2.pcolormesh(X, Y, V1)

fig3, ax3 = plt.subplots()
ax3.pcolormesh(X, Y, V2)
'''
plt.show()

temp = open("data.txt", "w")
for newl in V1:
    string = ''
    for e in newl:
        string += str(e.real) + '\t'
    string += '\n'
    temp.write(string)
temp.close()
