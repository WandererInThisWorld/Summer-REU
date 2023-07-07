import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Discrete Fourier Transform Implementation
def dft(y):
    N = len(y)
    mat = np.zeros((N, N))
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = (i * j) % N
    mat = np.exp(mat*-1j*2*np.pi/N)
    four = np.matmul(mat, y)
    return four

x = np.linspace(-6,6,10000)
y = 3 * np.sin(5*x)# + np.random.rand(100)
#df = np.abs(dft(y))
ff = np.abs(fft(y))
xf = np.ones(len(ff))
xf = xf * 2*np.pi/len(xf)

for i in range(len(xf)):
    xf[i] *= i

fig, ax = plt.subplots(1, 2)
ax[0].plot(x, y)
ax[1].plot(xf, ff)
#ax[1].plot(xf, df)
plt.show()
print(np.abs(dft(np.array([8, 4, 8, 0]))))
