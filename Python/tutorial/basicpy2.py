# Basic Python = good, Python with libraries = gooder
import numpy as np
import matplotlib.pyplot as plt


# We can do some simple operations with the numpy library
'''
ray = np.array([1, 2, 3, 4])
zeros = np.zeros(5)
ones = np.ones(5)
id = np.eye(5)
print(ray)
print(zeros)
print(ones)
print(id)
'''

# There are some operations that make sense and some that make
# sense after needing to use it often in very particular cases
'''
ray = np.array([1, 2, 3, 4])
sum = ray + ray
print(sum)
wth = ray * ray
print(wth)
print("\n\n")

id = np.ones((4, 4))
print(id*ray)
'''

# For example, consider this
def D(f):
    dx = 0.01
    dy = 0.01
    p_dfx = lambda x, y : (f(x + dx, y) - f(x, y))/dx
    p_dfy = lambda x, y : (f(x, y + dy) - f(x, y))/dy
    func = lambda x, y : np.array([p_dfx(x, y), p_dfy(x,y)])
    return func

f = lambda x, y : x*x*y*y

p_df = D(f)(1, 1)
print(p_df)

p_ddf = D(D(f))(1, 1)
print(p_ddf)

p_dddf = D(D(D(f)))(1, 1)
print(p_dddf)

p_ddddf = D(D(D(D(f))))(1, 1)
print(p_ddddf)

temp = np.matmul(p_ddddf, np.array([1, 0]))
print(temp)

# This is how to use matplotlib
# plt.subplots() is a function that takes in no input or
# two inputs, ncol and nrow, and outputs two objects,
# fig and ax. fig is the window, the "screen" fig is on
# ax is where the data is plotted
'''
x = np.linspace(-5, 5, num=100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
'''

# Multiple plots, same window
'''
fig, axs =  plt.subplot(2, 2)
x = np.linespace(-5, 5, num=100)
y1 = x*x
y2 = np.sin(x)
y3 = np.cos(x)
y4 = y3 + y1
print(axs)
axs[0][0].plot(x, y1)
axs[0][1].plot(x, y2)
axs[1][0].plot(x, y3)
axs[1][1].plot(x, y4)
plt.show()
'''

# Multiple windows
'''
x = np.linspace(-5, 5, num=100)
fig, ax = plt.subplots()
y1 = x
ax.plot(x, y1)
fig, ax = plt.subplots()
y2 = x*x
ax.plot(x, y2)
plt.show()
'''

# Multiple datasets, same axis
'''
x = np.linspace(-5, 5, num=100)
fig, ax = plt.subplots()
y1 = x
y2 = x*x
ax.plot(x, y1, x, y2)
plt.show()
'''



