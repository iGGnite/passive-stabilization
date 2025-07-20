from spacecraft_body import *
from time import time


if __name__ == "__main__":
    satellite = Satellite(1.2, .4, .4, 1.2, .4,
                          np.array([0, 0, 0, 0]), None)
    # satellite.
    tic = time()
    for n in range(500):
        # satellite.velocity_vector_i = np.array([-3-n/1000, 1+n/1000, 0])
        satellite.panel_angles = np.array([n/10, n/10, n/10, n/10])
        print(satellite.shadow_area)
    toc = time()
    print((toc - tic)/n)

    # satellite.velocity_vector_i = np.array([-5, 4, 0])

    # test_panel = satellite.panels[0]
    # test_panel.projected_area()

    # print(area)
    #TODO/IN PROGRESS: Start work on particle-panel interaction
    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation
