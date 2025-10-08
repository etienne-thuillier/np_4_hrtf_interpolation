import os
import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import lebedev_rule
from models.spherical_barycentric_interpolator import make_delaunay
from utilities.utilities import lebedev_degree_2_order


def cluster_triangle_mesh_simplices(faces, vertices, cluster_seeds):
    """
    associate each triangle simplex from a dense triangulation (faces, vertices) to the closest point in a sparse point
    cloud on the unit sphere, hence clustering said triangles into clusters approximately centered around said points
    """

    faces_2_cluster = np.empty((faces.shape[0],), dtype=int)
    for i, face in enumerate(faces):
        centroid = vertices[face].mean(0)
        distances_to_seeds = np.sqrt(((centroid - cluster_seeds) ** 2).sum(-1))
        faces_2_cluster[i] = np.argmin(distances_to_seeds)

    return faces_2_cluster


def triangle_area(vertices, keepdims=False):
    v0 = vertices[..., 0, :]
    v1 = vertices[..., 1, :]
    v2 = vertices[..., 2, :]

    e1 = v1 - v0
    e2 = v2 - v0

    signed_area = np.cross(e1, e2, axis=-1) / 2

    return np.sqrt((signed_area ** 2).sum(axis=-1, keepdims=keepdims))


def compute_cluster_centroids(mesh, cluster_seeds):
    """ centroid of triangular simplex cluster """

    face_centroids = mesh['vertices'][mesh['faces']].mean(1)
    face_areas = triangle_area(mesh['vertices'][mesh['faces']], keepdims=True)
    faces_2_cluster = cluster_triangle_mesh_simplices(faces=mesh['faces'],
                                                      vertices=mesh['vertices'], cluster_seeds=cluster_seeds)

    centroids = np.empty_like(cluster_seeds)
    for i, _ in enumerate(cluster_seeds):
        take = faces_2_cluster == i
        centroids[i] = (face_centroids[take] * face_areas[take]).sum(axis=0)
        centroids[i] /= np.sqrt((centroids[i] ** 2).sum())

    return centroids


def center_cluster_seeds(mesh, cluster_seeds, plot_clustering, n_iterations=100):
    """
        algorithm that iteratively moves the cluster seeds towards their respective centroids of the corresponding
        spherical voronoi tesselation

        the resulting seeds are hence approximately distributed in a uniform fashion over the unit sphere
    """

    if plot_clustering:
        plt.figure()
        plt.subplot(1, 1, 1, projection='3d')

    for _ in range(n_iterations):

        centroids = compute_cluster_centroids(mesh, cluster_seeds)

        delta = centroids - cluster_seeds
        cluster_seeds += delta
        cluster_seeds /= np.sqrt((cluster_seeds ** 2).sum(-1, keepdims=True))

        if plot_clustering:

            ''' plot '''

            def plot_polygon(polygon):
                p = np.concatenate((polygon[[-1], :], polygon), axis=0)
                plt.gca().plot3D(p[:, 0], p[:, 1], p[:, 2])

            # reset color cycle
            plt.gca().set_prop_cycle(None)

            voronoi_sparse = scipy.spatial.SphericalVoronoi(points=cluster_seeds)
            voronoi_sparse.calculate_areas()  # TODO: replace with voronoi.sort_vertices_of_regions() and test

            for i, region in enumerate(voronoi_sparse.regions):
                plot_polygon(voronoi_sparse.vertices[region])
                plt.gca().scatter(cluster_seeds[i, 0], cluster_seeds[i, 1], cluster_seeds[i, 2])

    plt.show()

    return cluster_seeds


def make_nearly_uniform_s2_grids(destination_folder, n_datapoints, high_density_mesh, n_iterations,
                                 plot_clustering=False, single_pickle_file=True):
    """
        make masks in which the observed point locations according to an approximately uniform grid on the sphere

        saves them to disk
    """

    x_array = np.zeros((len(n_datapoints), np.max(n_datapoints), 3), dtype=float)
    for i, n in enumerate(n_datapoints):

        print('%i-points grid' % n)

        if n == 0:

            x = np.array([[0.0, 0.0, 0.0]], dtype=float)

        elif n == 1:

            x = np.array([[1.0, 0.0, 0.0]], dtype=float)

        elif n == 2:

            x = np.array([[1.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0]], dtype=float)

        elif n == 3:

            x = np.array([[1.0, 0.0, 0.0],
                          [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0],
                          [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0.0]], dtype=float)

        else:

            x = np.random.normal(loc=0, scale=1, size=(n, 3))
            x /= np.sqrt((x ** 2).sum(-1, keepdims=True))

            x = center_cluster_seeds(high_density_mesh, x, plot_clustering, n_iterations=n_iterations)

            assert not np.any(np.isnan(x))

        x_array[i, :x.shape[0], :] = x

        if not single_pickle_file:
            with open(os.path.join(destination_folder, '%i_datapoints.npy' % n), 'wb') as f:
                np.save(f, x)

    if single_pickle_file:
        file = os.path.join(destination_folder, 'uniform_s2_grids.pickle')
        print('writing to ' + file)
        with open(file, 'wb') as f:
            pickle.dump(x_array, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_lebedev_mesh(lebedev_degree):
    order = lebedev_degree_2_order(lebedev_degree)
    x, _ = lebedev_rule(order)
    x = x.transpose()

    voronoi = scipy.spatial.SphericalVoronoi(points=x)
    voronoi_areas = voronoi.calculate_areas()  # TODO: replace with voronoi.sort_vertices_of_regions() and test
    vertices, faces = make_delaunay(voronoi)

    return {'vertices': vertices, 'faces': faces}


if __name__ == "__main__":
    n_vertices = 5810
    high_density_mesh = make_lebedev_mesh(lebedev_degree=n_vertices)

    print('demoing/plotting the algo with toy example')
    make_nearly_uniform_s2_grids(destination_folder='/tmp',
                                 n_datapoints=[9, ],
                                 high_density_mesh=high_density_mesh,
                                 n_iterations=10,
                                 plot_clustering=True)

    print('producing the s2 grids for experiments')
    make_nearly_uniform_s2_grids(destination_folder='/tmp',
                                 n_datapoints=list(range(0, 101)),
                                 high_density_mesh=high_density_mesh,
                                 n_iterations=200,
                                 plot_clustering=False)
