import numpy as np


# Any point on the line can be found using:

n = 10
t_space = np.linspace(0, 1, n)

point1 = np.array([.02, .02, .02])
# human shoulder
# [0.02 0.02 0.02]
# human elbow
# [ 0.20666667 -0.15777778 -0.15777778]
point2 = np.array([0.20666667, -0.15777778, -0.15777778])

for t in t_space:
    point = point1 + t * (point2 - point1)

    print(point)