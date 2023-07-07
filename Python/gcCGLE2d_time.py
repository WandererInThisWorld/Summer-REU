import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

N = 256
length = 200

x = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
y = length*np.array([i for i in range(-int(N/2), int(N/2))])/N
X, Y = np.meshgrid(x, y)

U = np.exp(-((X-0)**2 + (Y-0)**2))# + np.exp(-((X+20)**2 + (Y+0)**2))
#U = np.exp(U) - 1
#U = np.cos(np.sqrt(X*X + Y*Y))
#U = np.ones(np.shape(X))

V = fft2(U)
alpha = -0.5
beta = 0
sigma = 10
omega = -0.4 * np.exp(-(X*X+Y*Y)/(2*sigma*sigma))
mu = 0.5
chi = 0.65*np.pi
h = 1/4

bk = [i for i in range(0, int(N/2))]
bk[len(bk):] = [0]
bk[len(bk):] = [i for i in range(int(-N/2) + 1, 0)]
kx, ky = np.meshgrid(np.array(bk)*2*np.pi/length, np.array(bk)*2*np.pi/length)
#kx = np.array(bk)/16
#ky = np.array(bk)/16


L = -(kx*kx + ky*ky)

E = np.exp(h*L)
E2 = np.exp(h*L/2)

M = 16
r = np.exp(1j * np.pi * ((np.array([i for i in range(1, M+1)]))-0.5)/M)

newL = np.array([L for i in range(0,M)])
newL = np.transpose(newL)
newr = np.array([r for i in range(0, N)])
LR = []
for i in range(len(newL)):
    LR.append(h*newL[i] + newr)
#LR = h*newL + newr
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


fig, ax = plt.subplots()
ax.axis('off')
plot = ax.imshow(U)

def init():
    return plot

t = time.time()
count = 0

def animate(i):
    global Z, V, U, E2, Q, E, f1, f2, f3, t, count

    t0 = time.time()
    count += 1
    print(t - t0, '\t', count)
    t = t0

    Nv = fft2((1 - 1j*omega) * ifft2(V) - (1 + 1j*alpha)*ifft2(V)*abs(ifft2(V))**2 - mu * np.exp(1j*chi) * V[0][0]/(length**2))
    a = E2*V + Q*Nv
    Na = fft2((1 - 1j*omega) * ifft2(a) - (1 + 1j*alpha)*ifft2(a)*abs(ifft2(a))**2 - mu * np.exp(1j*chi) * a[0][0]/(length**2))
    b = E2*V + Q*Na
    Nb = fft2((1 - 1j*omega) * ifft2(b) - (1 + 1j*alpha)*ifft2(b)*abs(ifft2(b))**2 - mu * np.exp(1j*chi) * b[0][0]/(length**2))
    c = E2*a + Q*(2*Nb-Nv)
    Nc = fft2((1 - 1j*omega) * ifft2(c) - (1 + 1j*alpha)*ifft2(c)*abs(ifft2(c))**2 - mu * np.exp(1j*chi) * c[0][0]/(length**2))
    V = E*V + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    U = ifft2(V).real

    plot = ax.imshow(U)
    return plot

anim = FuncAnimation(fig, animate, init_func=init, blit=False, save_count=500, cache_frame_data=False)
anim.save('gccgleonepeak.gif')
#plt.show()


temp = open("data.txt", "w")
for newl in U:
    string = ''
    for e in newl:
        string += str(e) + '\t'
    string += '\n'
    temp.write(string)
temp.close()
