import matplotlib.pyplot as plt
import mapbox_earcut as earcut
import pointpats
from spacecraft_body import *
from time import time
from collisions import *
from shapely import plotting
from matplotlib.patches import Polygon as MplPolygon

if __name__ == "__main__":
    satellite = Satellite(1, .5, .5, 1, .5,
                          np.array([30, 30, 30, 30]), None)
    # satellite.
    # satellite.visualise()
    satellite.velocity_vector_i = np.array([-5, 4, 20])
    fig, ax = plt.subplots()
    plotting.plot_polygon(satellite.shadow)
    for tri in satellite.shadow_triangle_coords:
        patch = MplPolygon(tri, closed=True, edgecolor='black', alpha=0.3)
        ax.add_patch(patch)
    plt.show()
    # tic = time()
    # for n in range(500):
    #     # satellite.velocity_vector_i = np.array([-3-n/1000, 1+n/1000, 0])
    #     satellite.panel_angles = np.array([n/10, n/10, n/10, n/10])
    #     print(satellite.shadow_area)
    # toc = time()
    # print(f"{n+1} runs took {(toc - tic) / n} seconds")




    #TODO/IN PROGRESS: Start work on particle-panel interaction
    #TODO: Implement some function to calculate/approximate panel moments of inertia for later dynamics implementation
