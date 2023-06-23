import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib as mlp

N = 128

x = 32*np.pi*np.array([i for i in range(1, N + 1)])/N
u = np.cos(x/16)
v = fft(u)
alpha = 1 - 1j
beta = -1 + 0.5j
h = 1/4

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]

k = np.array(bk)/16
L = -k*k + alpha
E = np.exp(h*L)
E2 = np.exp(h*L/2)
print(L)

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


# main loop
uu = [u]
tt = [0]
tmax = 10
nmax = np.round(tmax/h)
nplt = 1#np.floor((tmax/100)/h)

for n in range(1, int(nmax) + 1):
    g = beta #-1j*fft(alpha * u + beta * abs(u)**2 * u)#-0.5j*k
    t = n*h
    Nv = g*fft(abs(ifft(v))**2)
    a = E2*v + Q*Nv
    Na = g*fft(abs(ifft(a))**2)
    b = E2*v + Q*Na
    Nb = g*fft(abs(ifft(b))**2)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = g*fft(abs(ifft(c))**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    u = ifft(v).real

    uu.append(u)
    tt.append(t)


fintt, finx = np.meshgrid(tt, x)
uu = np.array(uu).T


fig, ax = plt.subplots()
ax.pcolormesh(finx, fintt, uu)
plt.show()

print(np.shape(fintt))
print(np.shape(finx))
print(np.shape(uu))

temp = open("data.txt", "w")
for newl in uu:
    string = ''
    for e in newl:
        string += str(e) + '\t'
    string += '\n'
    temp.write(string)
temp.close()
