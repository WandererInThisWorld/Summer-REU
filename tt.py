import re
import numpy as np
'''
read = open("data.txt", "r")
count = 0
for line in read.readlines():
    count -= float(line.split(' ')[0])
    print(-float(line.split(' ')[0]))
print(count/60)

'''

a = 2
b = 0.2
sigma = (4/3)*(1/np.sqrt(a))

b1 = -0.1
b2 = 0.1
c1 = 0
c2 = 0.5
d2 = 0
d3 = 0

alpha = (4/3)*(1/np.sqrt(a*sigma))
beta = d2/(2*np.sqrt(a*sigma))
w = ((-sigma*np.sqrt(a*sigma))/(2*np.sqrt(3)))*(b1*a-b2/sigma)
dw = ((-sigma*np.sqrt(a*sigma))/(2*np.sqrt(3)))*(c2/sigma-c1*a)
mu = abs(((1j*sigma*np.sqrt(a*sigma))/(2*np.sqrt(3))) * (b2/sigma - b1*a))
chi = np.log((((1j*sigma*np.sqrt(a*sigma))/(2*np.sqrt(3))) * (b2/sigma - b1*a)) / mu)

print(alpha)
print(beta)
print(w)
print(dw)
print(mu)
print(chi)
