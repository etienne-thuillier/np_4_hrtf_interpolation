""" barycentric interpolator baseline (without any learnable parameters) wrapper to allow producing performance metrics
	from the same pipeline used for testing the neural process interpolator models """

import numpy as np
import scipy
import flax.linen as nn
from functools import partial
from jax.tree_util import tree_map
import jax.numpy as jnp


def make_delaunay(voronoi):
    """ ... """

    ''' cycle through the vertices of the vonoroi tessalation, creating a face for each vertex '''

    faces = np.empty((voronoi.vertices.shape[0], 3), dtype=int)

    for i, vertex in enumerate(voronoi.vertices):

        region_indices = [j for j in range(len(voronoi.regions)) if i in voronoi.regions[j]]

        assert len(region_indices) == 3

        ''' each region corresponds to a vonoroi points/seed, which forms a vertex in the Delaunay triangulation '''
        faces[i, :] = np.array(region_indices)

    return voronoi.points, faces


def order_vertices(vertices, faces):

    for i, face in enumerate(faces):

        v = vertices[face]
        center = np.mean(v, axis=0)

        e1 = v[1] - v[0]
        e2 = v[2] - v[0]

        cross = np.cross(e1, e2)

        if np.dot(center, cross) < 0:

            face = np.array([face[1], face[0], face[2]], dtype=int)

            faces[i, :] = face

    return faces


def global_2_local_coordinates(v0, v1, v2, x):
    """ ... """

    ''' edges and column space defined by said edges '''
    e0 = v1 - v0
    e1 = v2 - v0
    E = np.stack((e0, e1), axis=-1)

    ''' projection of target location on column space (normal equation) '''
    b = E.transpose().dot(x - v0)
    A = E.transpose().dot(E)

    xi = np.linalg.solve(A, b)

    return xi


def local_2_global_coordinates(v0, v1, v2, xi):

    e0 = v1 - v0
    e1 = v2 - v0
    E = np.stack((e0, e1), axis=-1)

    return E.dot(xi) + v0


def s2_2_mesh_location(vertices, faces, x_euclidian_on_S2):

    closest_face = get_closest_face({'vertices':vertices, 'faces': faces}, x_euclidian_on_S2)

    v0 = vertices[faces[closest_face]][0, :]
    v1 = vertices[faces[closest_face]][1, :]
    v2 = vertices[faces[closest_face]][2, :]

    xi = global_2_local_coordinates(v0=v0, v1=v1, v2=v2, x=x_euclidian_on_S2)

    x_euclidian_on_mesh = local_2_global_coordinates(v0=v0, v1=v1, v2=v2, xi=xi)

    return closest_face, x_euclidian_on_mesh


def make_subsampled_mesh(unit_sphere_locations):

    # voronoi = scipy.spatial.SphericalVoronoi(points=points[keep, :])
    voronoi = scipy.spatial.SphericalVoronoi(points=unit_sphere_locations)
    voronoi.calculate_areas()

    vertices, faces = make_delaunay(voronoi)
    faces = order_vertices(vertices, faces)

    assert np.all(vertices == unit_sphere_locations)

    return vertices, faces


def get_closest_face(mesh, x_euclidian_on_S2):

    vertices = mesh['vertices']
    faces = mesh['faces']

    squared_distance = ((x_euclidian_on_S2 - vertices[faces, ...].mean(1)) ** 2).sum(1)
    closest_face = np.argmin(squared_distance)

    return closest_face


def compute_S2_angles(v0, v1, v2):

    eps = np.finfo(dtype=v0.dtype).eps

    n01 = np.cross(v0, v1)
    n02 = np.cross(v0, v2)
    n12 = np.cross(v1, v2)

    n01 /= np.sqrt((n01 ** 2).sum()) + eps
    n02 /= np.sqrt((n02 ** 2).sum()) + eps
    n12 /= np.sqrt((n12 ** 2).sum()) + eps

    def theta(n1, n2):

        if np.all(n1 == 0) or np.all(n2 == 0):

            return 0

        else:

            return np.arccos(np.clip(n1.dot(n2), a_min=-1, a_max=1))

    theta0 = theta(n01, n02)
    theta1 = theta(-n01, n12)
    theta2 = theta(-n12, -n02)

    return theta0, theta1, theta2


def compute_S2_triangular_area(v0, v1, v2):
    """ computes the unit sphere area of a triangle on the surface with vertices v0, v1, v2 (all of which assumed on
        the unit sphere """

    if np.all(v0 == v1) or np.all(v1 == v2) or np.all(v2 == v0):

        return 0

    else:

        theta0, theta1, theta2 = compute_S2_angles(v0, v1, v2)

        area = theta0 + theta1 + theta2 - np.pi

        # import pdb
        # if area <= -1.0e-6:
        # 	pdb.set_trace()

        assert area > 0

        # area = np.clip(area, a_min=0.0, a_max=None)

        return area


def compute_barycentric_coordinates_v2(v0, v1, v2, x):
    """
        barycentric coordinates on a triangular slice of the S2 sphere
    """

    a2 = compute_S2_triangular_area(x, v0, v1)
    a1 = compute_S2_triangular_area(x, v0, v2)
    a0 = compute_S2_triangular_area(x, v1, v2)

    tot = a0 + a1 + a2

    return a0/tot, a1/tot, a2/tot


def compute_barycentric_coordinates(v0, v1, v2, x):
    """
        barycentric coordinates on a triangle simplex
        https://www.cdsimpson.net/2014/10/barycentric-coordinates.html
    """

    e0 = v0 - x
    e1 = v1 - x
    e2 = v2 - x

    a2 = np.cross(e0, e1) / 2
    a1 = np.cross(e0, e2) / 2
    a0 = np.cross(e1, e2) / 2

    a = lambda a: np.sqrt((a ** 2).sum())
    a0 = a(a0)
    a1 = a(a1)
    a2 = a(a2)
    tot = a0 + a1 + a2

    return a0/tot, a1/tot, a2/tot


def naive_spherical_barycentric_interpolation(mesh, values, target_locations):
    """ simple version of barycentric interpolation in which the interpolation is carried out, for each target, on the
        surface of the sphere's mesh simplex which 'contains' said target """

    vertices = mesh['vertices']
    faces = mesh['faces']

    v_shape = values.shape[1:]
    n, _ = target_locations.shape
    interpolated_values = np.empty(shape=(n, *v_shape), dtype=values.dtype)
    for i, target_location in enumerate(target_locations):

        face, x_euclidian_on_simplex = s2_2_mesh_location(vertices, faces, target_location)

        w0, w1, w2 = compute_barycentric_coordinates(v0=vertices[faces[face]][0, :],
                                                     v1=vertices[faces[face]][1, :],
                                                     v2=vertices[faces[face]][2, :],
                                                     x=x_euclidian_on_simplex)

        interpolated_values[i] = values[faces[face]].transpose().dot(np.stack((w0, w1, w2), axis=0)).transpose()

    return interpolated_values


def spherical_barycentric_interpolation(mesh, values, target_locations):
    """ improved version of barycentric interpolation in which the interpolation is carried out along the
        surface of the sphere (instead of the flat surface of the triangle simplex)
    """

    vertices = mesh['vertices']
    faces = mesh['faces']

    v_shape = values.shape[1:]
    n, _ = target_locations.shape
    interpolated_values = np.empty(shape=(n, *v_shape), dtype=values.dtype)
    for i, target_location in enumerate(target_locations):

        face = get_closest_face(mesh=mesh, x_euclidian_on_S2=target_location)

        w0, w1, w2 = compute_barycentric_coordinates_v2(v0=vertices[faces[face]][0, :],
                                                        v1=vertices[faces[face]][1, :],
                                                        v2=vertices[faces[face]][2, :],
                                                        x=target_location)

        interpolated_values[i] = values[faces[face]].transpose().dot(np.stack((w0, w1, w2), axis=0)).transpose()

    return interpolated_values


class BarycentricInterpolator(nn.Module):
    sigma_value: float

    @nn.compact
    def __call__(self, x_c, y_c, x_t, mask, *args, **kwargs):

        assert len(x_t.shape) in [3, 4], 'expected array of shape (batch_size, context_data_points, 3) or ' \
                                         '(batch_size, resolution, resolution, 3)'
        assert len(x_c.shape) in [3, 4], 'expected array of shape (batch_size, context_data_points, 3) or ' \
                                         '(batch_size, resolution, resolution, 3)'

        @partial(partial, tree_map)
        def check_y(y):

            if mask is not None:

                y.shape[1] == mask.shape[1]

            assert np.prod(y.shape[1:-4]) == np.prod(x_c.shape[1:-1]), 'spatial dimension(s) of context ' \
                                                                       'datapoint pair (x_c, y_c) must agree'

            assert len(y.shape) in [6, 7], 'expected array of shape (batch_size, context_data_points, sequence, spins,' \
                                           'channels, real/imaginary part) or (batch_size, resolution, resolution, sequence,' \
                                           'spins, channels, real/imaginary part), where channel is typically the ear index'

            assert y.shape[-1] in [1, 2], 'expected trailing dimension to be 1 (real input) or 2 (complex input)'

        check_y(y_c)

        ''' number of channels for which we need to predict mu_t and sigma_t '''

        if mask is None:

            assert False, 'never tested this case, put breakpoint here and check if makes sense'
            mask = jnp.ones((1, x_c.shape[1], 1, 1, 1, 1), dtype=x_c.dtype)

        def interpolate(x, y, x_target, m):
            """ interpolation for single batch element """

            m = np.squeeze(m)

            assert len(m.shape) == 1, 'assuming here that mask sampled only in spatial dimension, in particular not in the sequence/frequency dimension'

            indices = np.where(m)[0]

            if indices.shape[0] < 4:
                # can't interpolate with fewer than 4 context points: return 0 valued prediction
                y_target = jnp.zeros((x.shape[0], *y.shape[1:]), dtype=float)

            else:

                y = y[indices, ...]
                x = x[indices, ...]

                ss_vertices, ss_faces = make_subsampled_mesh(unit_sphere_locations=x)

                y_target = spherical_barycentric_interpolation(mesh={'vertices': ss_vertices, 'faces': ss_faces},
                                                               values=y,
                                                               target_locations=x_target)

            return y_target

        @partial(partial, tree_map)
        def interpolate__multimap_wrapper(y):

            mu_t = list()
            for i in range(y.shape[0]):

                mu_t += [interpolate(x_c[i], y[i], x_t[i], mask[i])]

            mu_t = np.stack(mu_t, axis=0)

            ''' adding sample lead dimension '''

            return np.expand_dims(mu_t, axis=0)

        mu_t = interpolate__multimap_wrapper(y_c)
        sigma_t = tree_map(lambda z: self.sigma_value * np.ones_like(z)[..., :1], mu_t)

        ''' adding sample lead dimension '''

        return mu_t, sigma_t


if __name__ == "__main__":

    compute_S2_angles(v0=np.array([0, 0, 1], dtype=float),
                      v1=np.array([0, 1, 0], dtype=float),
                      v2=np.array([0, 0, 1], dtype=float))
