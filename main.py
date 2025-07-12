from spacecraft_body import *



if __name__ == "__main__":
    satellite = Satellite(1, .3, .3, .8, .3,
                          np.array([30, 30, 30, 30]), None)
    satellite.print_nodes()
    #TODO: Include angles satellite panels in body creation (requires different position vector)
    #TODO: Think up some way to graphically represent the spacecraft body to visually confirm geometry
    #TODO: Start work on particle-panel interaction