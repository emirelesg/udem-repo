#!/usr/bin/python3
# https://share.cocalc.com/share/7557a5ac1c870f1ec8f01271959b16b49df9d087/Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb?viewer=share


from scipy.linalg import block_diag
import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
class Tracker():

    def __init__(self):


        dt = 1.0 / 20.0

        self.kalman = KalmanFilter(dim_x=4, dim_z=2)

        # x : ndarray (dim_x, 1), default = [0,0,0…0]
        #     Initial filter state estimate
        self.kalman.x = np.array([
            [0],
            [0],
            [0],
            [0]
        ])

        # P : ndarray (dim_x, dim_x), default eye(dim_x)
        #     covariance matrix
        uncertaintyInit = 500
        self.kalman.P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) * uncertaintyInit

        # Q : ndarray (dim_x, dim_x), default eye(dim_x)
        #     Process uncertainty/noise
        processVariance = 30
        q = Q_discrete_white_noise(dim=2, dt=dt, var=processVariance)
        self.kalman.Q = block_diag(q, q)
        print(self.kalman.Q)
        
        # R : ndarray (dim_z, dim_z), default eye(dim_x)
        #     measurement uncertainty/noise
        self.kalman.R = np.array([
            [0.5, 0],
            [0, 0.5]
        ])
        
        # H : ndarray (dim_z, dim_x)
        #     measurement function
        self.kalman.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # F : ndarray (dim_x, dim_x)
        #     state transistion matrix
        self.kalman.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # B : ndarray (dim_x, dim_u), default 0
        #     control transition matrix 


    def add(self, poly):
        
        self.kalman.predict()
        
        if (poly):
            m = poly.coeffs[0]
            b = poly.coeffs[1]
            measurement = np.array([
                [m],
                [b]
            ], np.float32)
            self.kalman.update(measurement)

        line = np.poly1d([self.kalman.x[0, 0], self.kalman.x[2, 0]])
        return line