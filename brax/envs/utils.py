import numpy as np


# Any point on the line can be found using:

n = 5
t_space = np.linspace(0, 1, n)

# human shoulder
shoulder = [.02, .02, .02]
elbow = [0.20666667, -0.15777778, -0.15777778]
larm = [-0.005, .025, .03]
hand = [.15, .18, .2]


point1 = np.array(larm)
point2 = np.array(hand)

for t in t_space:
    point = point1 + t * (point2 - point1)

    print(point)