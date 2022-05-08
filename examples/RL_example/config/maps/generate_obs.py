import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from math import ceil, floor
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
image = plt.imread("./levine_block.png")
original_image = copy.deepcopy(image)


def draw_rect(im, x, y, size):
    # Draws a rectangle, interpolating to allow floating point coords
    s = int(size / 2.)
    im[ceil(x) - s:floor(x) + s, ceil(y) - s:floor(y) + s] = 2.0
    return im


def GetCross(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])


def IsPointIn_outer_Matrix(p):
    p1 = np.array([45., 420. - 125.])
    p2 = np.array([52., 420. - 345.])
    p3 = np.array([552., 420. - 320])
    p4 = np.array([543., 420. - 112])
    return GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0


def IsPointIn_inner_Matrix(p):
    p1 = np.array([78., 420. - 160.])
    p2 = np.array([84., 420. - 306.])
    p3 = np.array([519., 420. - 285.])
    p4 = np.array([513., 420. - 147.])
    return GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0


def IsPointInMatrix(p, p_):
    p_[:, 1] = 420. - p_[:, 1]
    return GetCross(p_[0], p_[1], p) * GetCross(p_[2], p_[3], p) >= 0 and GetCross(p_[1], p_[2], p) * GetCross(p_[3],
                                                                                                               p_[0],
                                                                                                               p) >= 0


import random


def check_distance(p_list, min_len):
    # print("p_list", p_list.shape)
    num_object = p_list.shape[0]
    p = p_list.reshape(p_list.shape[0], 2)
    for i in range(num_object):
        p_toCompare = np.repeat(p_list[i].reshape(1, 2, 1), num_object, axis=0).reshape(num_object, 2)
        length = np.linalg.norm(p_list - p_toCompare, axis=1)
        c = np.where(length < min_len, 1, 0)
        if np.count_nonzero(c) > 1:
            return True
    return False
    # print(p_toCompare.shape,p_toCompare)


def random_point(type_):
    if type_ == "left":
        p = np.zeros([2, ])
        p[0] = random.uniform(55., 78.)
        p[1] = random.uniform(165., 300.)
    if type_ == "right":
        p = np.zeros([2, ])
        p[0] = random.uniform(522., 540.)
        p[1] = random.uniform(154., 279.)
    if type_ == "bottom":
        p = np.zeros([2, ])
        p[0] = random.uniform(90., 510.)
        p[1] = random.uniform(300., 350.)
    if type_ == "top":
        p = np.zeros([2, ])
        p[0] = random.uniform(90., 510.)
        p[1] = random.uniform(129., 147.)
    return p


def set_obstacle(img, square, object_num, type_):
    check_ = True
    image_ = copy.deepcopy(img)
    while check_:
        check_ = True
        p = random_point(type_)
        copied_p = copy.deepcopy(p)
        check_ = check_ and IsPointInMatrix(copied_p, square)
        # plt.imshow(image, cmap = 'gray')
        # plt.plot(p[0], p[1], 'rp', markersize = 10)
        # plt.show()
        reverse_image = np.where(image_ < 0.5, 1., 0.)
        ob_s = random.uniform(8., 12.)
        tmp_image = draw_rect(reverse_image, p[1], p[0], ob_s)
        reverse_image1 = np.where(image < 0.5, 1., 0.)
        t = np.unique(tmp_image + reverse_image1)
        # print(np.max(t))
        if np.max(t) < 3.0 and check_:
            check_ = True
            image_ = np.where(tmp_image < 0.5, 1., 0.)
            # print("check", check_)
        else:
            check_ = False
        # if check_==True:
        #     check_=check_distance(p_np)
        check_ = not check_
    return image_, p


def each_part_set_obstacle(image, square, num, min_length, type_):
    check_ = True
    image_temp = copy.deepcopy(image)
    while check_:
        p_array = []
        for i in range(num):
            if i == 0:
                image_, p = set_obstacle(image_temp, square, 1, type_=type_)
                p_array.append(p)
            else:
                image_, p = set_obstacle(image_, square, 1, type_=type_)
                p_array.append(p)
        check_ = check_distance(np.array(p_array), min_length)
        print(p_array)
    return image_


# print(image.shape)
image = np.where(image < 0.5, 0., 1.0)
# print(image)
# plt.imshow(image, cmap='gray')
square_l = np.zeros((4, 2))
square_l[0] = np.array([45., 160.])
square_l[1] = np.array([50., 306.])
square_l[2] = np.array([84., 306.])
square_l[3] = np.array([78., 160.])
square_r = np.zeros((4, 2))
square_r[0] = np.array([513., 147.])
square_r[1] = np.array([519., 285.])
square_r[2] = np.array([550., 285.])
square_r[3] = np.array([545., 147.])
square_b = np.zeros((4, 2))
square_b[0] = np.array([84., 306.])
square_b[1] = np.array([84., 337.])
square_b[2] = np.array([519., 318.])
square_b[3] = np.array([519., 285.])
plt.plot(square_b[0, 0], square_b[0, 1], 'rp', markersize=5)
plt.plot(square_b[1, 0], square_b[1, 1], 'rp', markersize=5)
plt.plot(square_b[2, 0], square_b[2, 1], 'rp', markersize=5)
plt.plot(square_b[3, 0], square_b[3, 1], 'rp', markersize=5)
square_t = np.zeros((4, 2))
square_t[0] = np.array([78., 129.])
square_t[1] = np.array([78., 160.])
square_t[2] = np.array([513., 147.])
square_t[3] = np.array([513., 115.])
# plt.show()
parser = argparse.ArgumentParser()
parser.add_argument("--num_long_size", type=int, default=4)
parser.add_argument("--num_short_size", type=int, default=2)
parser.add_argument("--min_length", type=float, default=40.)
args = parser.parse_args()
object_num_long = 2
num_long_size = args.num_long_size
num_short_size = args.num_short_size
image_ = each_part_set_obstacle(image, square_t, num_long_size, args.min_length, "top")
image_ = each_part_set_obstacle(image_, square_b, num_long_size, args.min_length, "bottom")
image_ = each_part_set_obstacle(image_, square_l, num_short_size, args.min_length, "left")
image_ = each_part_set_obstacle(image_, square_r, num_short_size, args.min_length, "right")

image_ = image_ + original_image - 1.0
import cv2
cv2.imshow('image_', image_)
cv2.waitKey(0)
