"""Implementations for computing the exact hypervolume indicator."""
import numpy as np

from anguilla.ds.rbtree import RBTree


def calculate_2d(ps: np.ndarray, ref_p: np.ndarray) -> float:
    """Compute the exact 2D hypervolume indicator.

    Parameters
    ----------
    ps
        The point set.
    ref_p
        The reference point.

    Notes
    -----
    Ported from :cite:`2008:shark`.

    """
    if ps.shape[0] == 0:
        return 0.0

    # Copy point set and sort along its first dimension.
    sorted_idx = np.argsort(ps[:, 0])
    ps = ps[sorted_idx]

    # Perform the integration.
    volume = (ref_p[0] - ps[0][0]) * (ref_p[0] - ps[0][1])

    last_valid_idx = 0
    for k in range(1, ps.shape[0]):
        dim1_diff = ps[last_valid_idx][1] - ps[k][1]
        # skip dominated points
        # point is dominated <=> dim1_diff <= 0
        if dim1_diff > 0:
            volume += (ref_p[0] - ps[k][0]) * dim1_diff
            last_valid_idx = k

    return volume


def calculate_3d(ps: np.ndarray, ref_p: np.ndarray) -> float:
    """Calculate the exact 3D hypervolume indicator.

    Parameters
    ----------
        ps
            The point set.
        ref_p
            The reference point.

    Notes
    -----
    Implements Algorithm 1 from :cite:`2009-hypervolume-hv3d`, \
    but with the difference of assuming a minimization problem. \
    See also algorithm HV3D presented in :cite:`2020:hypervolume` and the \
    explanation from sec. 4.1 of p. 6. of :cite:`2009-hypervolume-hv3d`.

    The differences in the implementation, w.r.t. the algorithm description \
    in the paper, are primarily due to them assuming a maximization \
    problem and the implementation the opposite (minimization).

    The following figure shows an example of how the algorithm is \
    transformed for working with a minimization problem:

    .. image:: /figures/hv3d_min.png
       :width: 750
       :alt: Example of the algorithm for minimization problem.
    """
    if ps.shape[0] == 0:
        return 0.0

    # Filter out any points that are dominated by the reference point
    nondominated_idx = np.all(ps < ref_p, axis=1)
    nondominated_ps = ps[nondominated_idx]

    if nondominated_ps.shape[0] == 0:
        return 0.0

    # Sort the points by their z-coordinate in ascending order.
    sorted_idx = np.argsort(nondominated_ps[:, 2])
    sorted_ps = nondominated_ps[sorted_idx]

    # TODO: replace RBT with an AVL tree?

    # The algorithm works by performing sweeping in the z-axis,
    # and it uses a tree with balanced height as its sweeping structure
    # (e.g. an AVL tree) in which the keys are the x-coordinates and the
    # values are the y-coordinates.
    # Currently we use a red-black tree (as Shark uses std::map).
    # As in p. 6 of [2009:hypervolume-hv3d], the structure mantains the
    # x-coordinates in ascending order.
    front_2d = RBTree()

    # As explained in [2009:hypervolume-hv3d], we use two sentinel points
    # to ease the handling of boundary cases by ensuring that succ(p_x)
    # and pred(p_x) are defined for any other p_x in the tree.
    ref_x, ref_y, ref_z = ref_p

    front_2d.insert(ref_x, float("-inf"))
    front_2d.insert(float("-inf"), ref_y)

    # The first point from the set is added.
    p_x, p_y, p_z = sorted_ps[0]
    front_2d.insert(p_x, p_y)
    last_z = p_z  # the highest minimial point seen so far

    area = (ref_x - p_x) * (ref_y - p_y)
    volume = 0.0

    # And then all the other points are processed.
    for p_x, p_y, p_z in sorted_ps[1:, :]:
        # find greatest q_x, such that q_x <= p_x
        node_q = front_2d.lower_bound_by_key(p_x)
        q_x, q_y = node_q

        if p_y < q_y:  # p is non-dominated by q

            volume += area * (p_z - last_z)
            last_z = p_z

            # remove dominated points and their area contributions
            prev_x = p_x
            prev_y = q_y
            node_s = front_2d.succ(node_q)
            s_x, s_y = node_s

            while True:
                area -= (s_x - prev_x) * (ref_y - prev_y)
                if p_y > s_y:  # guaranteed by the sentinel point
                    break
                prev_x = s_x
                prev_y = s_y
                dominated_node = node_s
                node_s = front_2d.succ(node_s)
                s_x, s_y = node_s
                front_2d.remove(dominated_node)

            # add the new point
            area += (s_x - p_x) * (ref_y - p_y)
            front_2d.insert(p_x, p_y)

    # Add last point's contribution to the volume
    volume += area * (ref_z - last_z)

    return volume
