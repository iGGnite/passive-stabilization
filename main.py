import mapbox_earcut as earcut
import pointpats
from spacecraft_body import *
from time import time
from collisions import *
from shapely import plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    # np.random.seed(12)
    satellite = Satellite(1, .5, .5, 1, .5,
                          np.array([60, 60, 60, 60]), None)
    # satellite.
    print()
    satellite.velocity_vector_i = np.array([-6, 3, 12])
    _, p3d = satellite.get_random_point()
    impacts = satellite.get_distance_to_planes(p3d,satellite.velocity_vector_b)
    random_2d_points = []
    random_3d_points = []
    for k in range(4):
        p2d, p3d = satellite.get_random_point()
        random_2d_points.append(p2d)
        random_3d_points.append(p3d)
    fig, ax = plt.subplots()
    plotting.plot_polygon(satellite.shadow)
    for tri in satellite.shadow_triangle_coords:
        patch = MplPolygon(tri, closed=True, edgecolor='black', alpha=0.3)
        ax.add_patch(patch)
    for point in random_2d_points:
        ax.scatter(point[0], point[1], color='red')
    basis = satellite.shadow_projection_axis_system
    x1 = np.dot(satellite.body_forward_vectors[0], basis[0])
    y1 = np.dot(satellite.body_forward_vectors[1], basis[1])

    x2 = np.dot(satellite.body_normal_vectors[0], basis[0])
    y2 = np.dot(satellite.body_normal_vectors[1], basis[1])
    ax.quiver(x1, y1, color='green')
    ax.quiver(x2, y2, color='blue')
    satellite.visualise(particle_vectors=random_3d_points)
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
