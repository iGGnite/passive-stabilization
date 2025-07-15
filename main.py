from spacecraft_body import *



if __name__ == "__main__":
    satellite = Satellite(1.2, .4, .4, 1, .4,
                          np.array([10, 10, 40, 40]), None)
    satellite.print_nodes()
    satellite.visualise()
    #TODO: Include angles satellite panels in body creation (requires different position vector)
    #TODO: Think up some way to graphically represent the spacecraft body to visually confirm geometry
    #TODO: Start work on particle-panel interaction
    #TODO: Implement some function to calculate panel moments of inertia