from spacecraft_body import *


if __name__ == "__main__":
    satellite = Satellite(3, 1, 1, 1, .4,
                          np.array([25, 25, 25, 25]), None)
    print(satellite.normal_plane_velocity_vector)
    satellite.velocity_vector_i = np.array([-3, 1, 0])
    print(satellite.normal_plane_velocity_vector)
    # test_panel = satellite.panels[0]
    # test_panel.projected_area()

    # print(area)
    #TODO/IN PROGRESS: Start work on particle-panel interaction
    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation