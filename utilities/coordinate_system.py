import numpy as np


def convert_2_common_coordiate_system(s):
    """ converts relevant 3D position quantities of sofa object to cartesian coordinate system """

    if s.ReceiverPosition_Type == 'cartesian' and s.ReceiverPosition_Units == 'metre':

        ear_positions = s.ReceiverPosition.squeeze()

    else:

        raise ValueError

    if s.SourcePosition_Type == 'spherical' and s.SourcePosition_Units == 'degree, degree, metre':

        x, y, z = sph2cart__matlab(azimuth=np.deg2rad(s.SourcePosition[:, 0]),
                                   elevation=np.deg2rad(s.SourcePosition[:, 1]),
                                   radius=s.SourcePosition[:, 2])

        source_locations = np.stack((x, y, z), axis=1)

    elif s.SourcePosition_Type == 'cartesian' and s.SourcePosition_Units == 'meter':

        source_locations = s.SourcePosition.squeeze()

    else:

        raise ValueError('(' + s.SourcePosition_Type + ', ' + s.SourcePosition_Units + ') not supported')

    if s.ListenerPosition_Type == 'cartesian' and s.ListenerPosition_Units == 'metre':

        listener_position = s.ListenerPosition.squeeze()

    else:

        raise ValueError

    if s.ListenerView_Type == 'cartesian' and s.ListenerView_Units == 'metre':

        listener_view = s.ListenerView.squeeze()

    else:

        raise ValueError

    listener_up = s.ListenerUp.squeeze()

    return ear_positions, source_locations, listener_position, listener_view, listener_up


def cart2sph__sfs(x, y, z):
    r"""Literal copy of sfs's cart2sph method

    -----------------------------

    Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\alpha \in [-pi, pi], \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates

    Returns
    -------
    alpha : float or `numpy.ndarray`
            Azimuth angle in radiants
    beta : float or `numpy.ndarray`
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / r)
    return alpha, beta, r


def sph2cart__sfs(alpha, beta, r):
    r"""Literal copy of sfs's sph2cart method

    Spherical to cartesian coordinate transform.

    .. math::

        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return x, y, z


def cart2sph__matlab(x, y, z):
    """
        equivalent of matlab verison of cartesian to spherical coordinates conversion
        https://www.mathworks.com/help/matlab/ref/cart2sph.html
    """

    # import sfs
    # azimuth, elevation, radius = sfs.util.cart2sph(x=x, y=y, z=z)
    azimuth, elevation, radius = cart2sph__sfs(x=x, y=y, z=z)

    azimuth = np.fmod(azimuth + 2 * np.pi, 2 * np.pi)
    elevation = - elevation + np.pi / 2

    return azimuth, elevation, radius


def sph2cart__matlab(azimuth, elevation, radius):
    """
        equivalent of matlab version of spherical to cartesian coordinates conversion
        https://www.mathworks.com/help/matlab/ref/sph2cart.html
    """

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    return x, y, z


def spherical__matlab_2_scipy(azimuth: object, elevation: object, radius: object) -> object:
    """
    converts spherical coordinates from matlab convention,
        see https://www.mathworks.com/help/matlab/ref/sph2cart.html,
    to scipy convention
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html
    """

    polar = np.fmod(-(elevation + np.pi / 2) + 2 * np.pi, np.pi)

    return azimuth, polar, radius


if __name__ == '__main__':
    pass
