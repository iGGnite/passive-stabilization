from data_processing import plot_angular_momentum
from spacecraft_body import *
from time import time
from simulation import *
from shapely import plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from dataclasses import *

if __name__ == "__main__":
    # np.random.seed(2)
    sim = PassiveStabilization()
    sim.create_cubesat(1.2, .4, .4, np.array([30, 30, 30, 30]))
    state = sim.run_simulation()
    plot_angular_momentum(state)
    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation
    #TODO: Straight-on particles with zero degree panels results in Polygon error. Investigate!