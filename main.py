from processing import *
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
    run = True
    if run:
        sim = PassiveStabilization()
        sim.create_cubesat(1, .4, .4, np.array([0, 0, 0, 0]))
        CubeSat = sim.sat
        sim.inertial_to_body_quat = eul_to_quat(np.deg2rad(np.array([0,0,10])))
        # CubeSat.visualise()
        # sim.show_attitude()
        # sim.visualise_3d(True)
        CubeSat.panel_angles = np.array([20, 20, 20, 20])

        # CubeSat.calculate_inertia()

        # sim.visualise_3d(True,test_array)
        # particles = sim.sat.generate_impacting_particle(n_particles=600)
        # print(particles[1])
        # sim.visualise_3d(True,)
    #     state = sim.run_simulation(False)
    #     with open('state.pickle', 'wb') as f:
    #         pkl.dump(state, f)
    #
    # with open('state.pickle', 'rb') as f:
    #     state = pkl.load(f)
    # plot_angular_rate(state)
    # plt.show()
    # plot_attitude(state,False)
    # plt.show()
    # plot_attitude(state,True)
    # plt.show()
    # plot_angular_momentum(state)
    # plt.show()
    # animate_rotations(state)


    #TODO: It might be interesting to build a control system which only uses the panel angles
    #TODO: Straight-on particles with zero degree panels results in Polygon error. Investigate!