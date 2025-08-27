from spacecraft_body import *
from time import time
from simulation import *
from shapely import plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np


if __name__ == "__main__":
    # np.random.seed(2)
    sim = PassiveStabilization()
    sim.create_cubesat(.8, .4, .4, np.array([40, 40, 40, 40]))
    sim.simulate_timestep(visualise_timestep=True)

    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation
