# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# INITIALIZATION OF Q, pi AND C MATRICES
# =============================================================================
def data_init(x_racetrack,y_racetrack,max_speed_limit):
    Q = np.random.rand(y_racetrack,x_racetrack,max_speed_limit,max_speed_limit,9)*400 - 500
    
    C = np.zeros((y_racetrack,x_racetrack,max_speed_limit,max_speed_limit,9))
    
    pi = np.zeros((y_racetrack,x_racetrack,max_speed_limit,max_speed_limit),dtype='int')
    
    return Q, C, pi


    
