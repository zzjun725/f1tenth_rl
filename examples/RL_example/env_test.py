# -- coding: utf-8 --
import matplotlib.pyplot as plt
import numpy as np

waypoints_file = './example_waypoints.csv'

with open(waypoints_file, encoding='utf-8') as f:
    waypoints = np.loadtxt(f, delimiter=';')
    x = waypoints[:, 1]
    y = waypoints[:, 2]
    print(x.shape)
    k = np.vstack([x, y]).T
    print(k.shape)
    plt.plot(k[:, 0], k[:, 1])
    plt.show()