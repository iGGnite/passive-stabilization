from data_processing import *
from dynamics_helper_functions import quat_to_CTM
from spacecraft_body import *
from time import time
from simulation import *
from shapely import plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pickle as pkl

if __name__ == "__main__":
    np.random.seed(2)
    run = False
    if run:
        sim = PassiveStabilization()
        sim.create_cubesat(1.2, .4, .4, np.array([30, 30, 30, 30]))
        sim.aero_to_body_quat = eul_to_quat(np.deg2rad(np.array([0,0,10])))
        # sim.show_attitude()
        # sim.visualise_3d(True)
        # sim.visualise_3d(True,test_array)
        # particles = sim.sat.generate_impacting_particle(n_particles=600)
        # print(particles[1])
        # sim.visualise_3d(True,)
        state = sim.run_simulation()
        with open('state.pickle', 'wb') as f:
            pkl.dump(state, f)

    with open('state.pickle', 'rb') as f:
        state = pkl.load(f)
    plot_angular_rate(state)
    plt.show()
    plot_attitude(state,False)
    plt.show()
    quat_to_CTM(state[:,4:8])
    animate_rotations(state)


    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation
    #TODO: Straight-on particles with zero degree panels results in Polygon error. Investigate!