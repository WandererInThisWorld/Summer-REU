import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

'''
def convolution(Z, x, y):
    up = (x + len(Z[x]) + 1) % len(Z[x])
    down = (x + len(Z[x]) - 1) % len(Z[x])
    right = (y + len(Z) + 1) % len(Z)
    left = (y + len(Z) - 1) % len(Z)

    count = 0

    if Z[up][left] == 1:
        count = count + 1
    if Z[up][y] == 1:
        count = count + 1
    if Z[up][right] == 1:
        count = count + 1

    if Z[x][left] == 1:
        count = count + 1
    if Z[x][right] == 1:
        count = count + 1

    if Z[down][left] == 1:
        count = count - 3
    if Z[down][y] == 1:
        count = count + 1
    if Z[down][right] == 1:
        count = count - 1

    return count

def I(t, x, y):
    if t % 20 == 0:
        return np.array([0, 0])
    else:
        large = 1000
        return np.array([large, large])

def update(Z, t):
    # the map has no boundaries
    newZ = np.zeros(np.shape(Z))
    
    for row in range(len(Z)):
        for col in range(len(Z[row])):
            idx_up = (row + len(Z[row]) + 1) % len(Z[row])
            idx_down = (row + len(Z[row]) - 1) % len(Z[row])
            idx_right = (col + len(Z) + 1) % len(Z)
            idx_left = (col + len(Z) - 1) % len(Z)
            
            if Z[row][col] == 0:
                count = convolution(Z, row, col)
                count2 = 0

                if count >= 1 or count2 >= 3:
                    newZ[row][col] = 1
            elif Z[row][col] == 1:
                newZ[row][col] = 2
            elif Z[row][col] == 2:
                newZ[row][col] = 3
            elif Z[row][col] == 3:
                newZ[row][col] = 0
            

    return newZ
'''

x = 50
y = 50
Z = np.zeros((x, y))

'''
for i in range(int(len(Z)/2), len(Z)):
    Z[25][i] = 1
'''
Z[10][10] = 1
Z[10][11] = 1
Z[11][10] = 1
Z[11][11] = 1
'''
Z[39][29] = 1
Z[10][11] = 1
Z[11][10] = 1
Z[11][11] = 1
'''
print(Z)

fig, ax = plt.subplots()
ax.axis('off')
plot = ax.imshow(Z)

print(plot)

def init():
    return plot

def animate(i):
    global Z
    Z = update(Z, i)
    plot = ax.imshow(Z)
    return plot

anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=False)
plt.show()

