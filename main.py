from collisions import projected_area
from spacecraft_body import *
from collisions import *


if __name__ == "__main__":
    satellite = Satellite(1.2, .4, .4, 1, .4,
                          np.array([25, 25, 25, 25]), None)
    satellite.print_nodes()
    # satellite.visualise()
    test_panel = satellite.panels[1]
    area = projected_area(test_panel, np.eye(3), np.array([-10, 1, 1]))
    #TODO/IN PROGRESS: Start work on particle-panel interaction
    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation