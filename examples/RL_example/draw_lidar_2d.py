# -- coding: utf-8 --
import torch
import numpy as np
import matplotlib.pyplot as plt


class LIDAR:
    def __init__(self, map_size, x_range=15, y_range=15):
        self.map_size = map_size
        self.map = np.zeros((self.map_size, self.map_size))

        self.lidar_dmin = 0
        self.lidar_dmax = x_range
        self.angle_min = -135
        self.angle_max = 135
        self.x_min, self.x_max = -x_range, x_range
        self.y_min, self.y_max = -y_range, y_range
        self.resolution = 1
        self.interpolate_or_not = False

    def rays2world(self, distance):
        # convert lidar scan distance to 2d locations in space
        angles = np.linspace(self.angle_min, self.angle_max, self.dimension) * np.pi / 180
        x = distance * np.cos(angles)
        y = distance * np.sin(angles)
        return x, y

    def grid_cell_from_xy(self, x, y):
        # convert 2d locations in space to 2d array coordinates
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)

        cell_indices = np.zeros((2, x.shape[0]), dtype='int')
        cell_indices[0, :] = np.floor((x - self.x_min) / self.resolution)
        cell_indices[1, :] = np.floor((y - self.y_min) / self.resolution)
        return cell_indices

    def interpolate(self, cell_indices):
        for i in range(cell_indices.shape[1] - 1):
            fill_x = np.linspace(cell_indices[1, i], cell_indices[1, i + 1], endpoint=False, dtype='int')
            fill_y = np.linspace(cell_indices[0, i], cell_indices[0, i + 1], endpoint=False, dtype='int')
            self.map[fill_x, fill_y] = 1

    def convert_scan2map(self, lidar_1d):
        self.map = np.zeros((self.map_size, self.map_size))
        self.distance = lidar_1d
        self.dimension = len(self.distance)

        x, y = self.rays2world(self.distance)
        cell_indices = self.grid_cell_from_xy(x, y)
        self.map[cell_indices[1, :], cell_indices[0, :]] = 1

        if self.interpolate_or_not:
            self.interpolate(cell_indices[:, :])
        return self.map.copy()
    # plt.imshow(self.map)
    # plt.show()


if __name__ == "__main__":
    lidar = torch.load('lidar.pt')
    print(lidar)
    # plt.plot(lidar)
    # plt.show()

    lidar_plot = LIDAR()
    lidar_plot.convert_scan2map(lidar)
