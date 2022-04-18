import imp
import numpy as np
from math import pi


class TrackingPlanner:
    def __init__(self, wp_path, wp_gap = 1, debug=True):
        # wp
        self.wp = []  # (n, 3)
        self.wp_path = wp_path
        self.wpNum = None
        self.wp_path = wp_path
        self.wpGapCounter = 0
        self.wpGapThres = wp_gap
        self.max_speed = 70
        self.speedScale = 1.0


        # PID for speed
        self.Increase_P = 1 / 5
        self.Decrease_P = 1 / 6
        self.P = 10
        self.targetSpeed = 0
    
    def load_wp(self):
        with open(self.wpt_path, encoding='utf-8') as f:
            self.waypoints = np.loadtxt(f, delimiter=';')
            self.wp= \
                np.vstack([self.waypoints[:, 1], self.waypoints[:, 2], self.waypoints[:, 5]]).T
    
    def pidAccel(self, diff, targetS=0, curS=0):
        a = self.P * diff
        if a > 0 :
            a = self.Increase_P * a
        else: 
            a = self.Decrease_P * a
        print(f'a: {np.round(a, 3)}')
        a = np.clip(a, -1.0, 1.0)
        print(f'a: {np.round(a, 3)}')
        return np.clip(a, -1.0, 1.0)
    
    
    def planning(self, pose, speed):
        """
        pose: (global_x, global_y, yaw) of the car
        speed: current speed of the car

        Return:
        steering_angle, accelation
        """
        raise NotImplementedError
        # return steering_angle, accelation


class PurePursuitPlanner(TrackingPlanner):
    def __init__(self, wp_path, debug=False):
        super().__init__(wp_path, wp_gap=0, debug=debug)
        # self.wp = []
        self.wp_path = wp_path
        self.minL = 0.6
        self.maxL = 2.0
        self.minP = 0.5
        self.maxP = 0.8
        self.interpScale = 20
        self.Pscale = 15
        self.Lscale = 15
        self.interp_P_scale = (self.maxP-self.minP) / self.Pscale
        self.interp_L_scale = (self.maxL-self.minL) / self.Lscale
        self.prev_error = 0
        self.D = 0.15 
        self.errthres = 0.1
        # self.load_Optimalwp()  # (3, n), n is the number of waypoints
        self.load_wp()
    
    def find_targetWp(self, cur_position, targetL):
        """
        cur_positon: (2, )
        return: cur_L, targetWp(2, ), targetV 
        """
        # ipdb.set_trace()

        wp_xyaxis = self.wp[:2]  # (2, n)
        dist = np.linalg.norm(wp_xyaxis-cur_position.reshape(2, 1), axis=0)
        nearst_idx = np.argmin(dist)
        nearst_point = wp_xyaxis[:, nearst_idx]

        segment_end = nearst_idx
        for i, point in enumerate(wp_xyaxis.T[nearst_idx:]):
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist > targetL:
                break
        segment_end += i
        interp_point = np.array([wp_xyaxis[0][segment_end], wp_xyaxis[1][segment_end]])
        # get interpolation
        # error = 0.1
        x_array = np.linspace(wp_xyaxis[0][segment_end-1], wp_xyaxis[0][segment_end], self.interpScale)
        y_array = np.linspace(wp_xyaxis[1][segment_end-1], wp_xyaxis[1][segment_end], self.interpScale)
        v_array = np.linspace(self.wp[2][segment_end-1], self.wp[2][segment_end], self.interpScale)
        xy_interp = np.vstack([x_array, y_array])
        dist_interp = np.linalg.norm(xy_interp-cur_position.reshape(2, 1), axis=0) - targetL
        i_interp = np.argmin(np.abs(dist_interp))
        interp_point = np.array([x_array[i_interp], y_array[i_interp]])
        interp_v = v_array[i_interp]
        cur_L = np.linalg.norm(cur_position-interp_point)
        # ipdb.set_trace()
        return cur_L, interp_point, interp_v, segment_end, nearst_point
    
    def planning(self, pose, speed):
        """
        pose: (global_x, global_y, yaw) of the car
        """
        targetL = speed * self.interp_L_scale + self.minL
        P = self.maxP - speed * self.interp_P_scale 
        
        yaw = pose[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, pose[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, pose[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])        
        
        # wp_xyaxis = self.wp[:2]
        cur_L, targetWp, targetV, segment_end, nearstP = self.find_targetWp(pose[:2], targetL)

        global2local = np.linalg.inv(local2global)
        nearstP_local = global2local @ np.array([nearstP[0], nearstP[1], 0, 1]) 
        cur_error = nearstP_local[1]

        offset = self.D * (cur_error - self.prev_error)
        self.prev_error = cur_error
        print(f'D_offset: {offset}')

        local_goalP = global2local @ np.array([targetWp[0], targetWp[1], 0, 1])
        gamma = 2*abs(local_goalP[1]) / (cur_L ** 2)
        if local_goalP[1] < 0:
            steering_angle = P * -gamma
        else:
            steering_angle = P * gamma
        steering_angle = np.clip(steering_angle+offset, -1.0, 1.0)
        # self.targetSpeed = targetV
        # diff = targetV - speed
        # acceleration = self.pidAccel(diff)

        return steering_angle, targetV, targetWp    
