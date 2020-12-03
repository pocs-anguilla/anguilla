"""Implementations for computing the exact hypervolume indicator."""
import numpy as np
from collections import deque
from typing import Deque, List, Optional
from anguilla.ds.rbtree import RBTree


def calculate_2d(ps: np.ndarray, ref_p: Optional[np.ndarray] = None) -> float:
    """Compute the exact 2D hypervolume indicator.

    Parameters
    ----------
    ps
        The point set of mutually non-dominated points.
    ref_p
        The reference point. Otherwise assumed to be the origin.

    Notes
    -----
    Ported from :cite:`2008:shark`.

    """
    if len(ps) == 0:
        return 0.0

    if ref_p is None:
        ref_x, ref_y = 0.0, 0.0
    else:
        ref_x, ref_y = ref_p

    # Copy point set and sort along its first dimension.
    sorted_idx = np.argsort(ps[:, 0])
    sorted_ps = ps[sorted_idx]

    # Perform the integration.
    p_x, p_y = sorted_ps[0]
    volume = (ref_x - p_x) * (ref_y - p_y)

    last_y = p_y
    for p_x, p_y in sorted_ps[1:]:
        y_diff = last_y - p_y
        # skip dominated points
        # point is dominated <=> y_diff <= 0
        if y_diff > 0:
            volume += (ref_x - p_x) * y_diff
            last_y = p_y
    return volume


def calculate_3d(ps: np.ndarray, ref_p: Optional[np.ndarray] = None) -> float:
    """Calculate the exact 3D hypervolume indicator.

    Parameters
    ----------
        ps
            The point set.
        ref_p
            The reference point. Otherwise assumed to be the origin.

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
    if len(ps) == 0:
        return 0.0

    if ref_p is None:
        ref_x, ref_y, ref_z = 0.0, 0.0, 0.0
    else:
        ref_x, ref_y, ref_z = ref_p

    # Sort the points by their z-coordinate in ascending order.
    sorted_idx = np.argsort(ps[:, 2])
    sorted_ps = ps[sorted_idx]

    # TODO: replace RBT with an AVL tree?

    # The algorithm works by performing sweeping in the z-axis,
    # and it uses a tree with balanced height as its sweeping structure
    # (e.g. an AVL tree) in which the keys are the x-coordinates and the
    # values are the y-coordinates.
    # Currently we use a red-black tree (as Shark uses std::map).
    # As in p. 6 of [2009:hypervolume-hv3d], the structure mantains the
    # x-coordinates in ascending order.
    frond_xy = RBTree()

    # As explained in [2009:hypervolume-hv3d], we use two sentinel points
    # to ease the handling of boundary cases by ensuring that succ(p_x)
    # and pred(p_x) are defined for any other p_x in the tree.
    frond_xy[ref_x] = float("-inf")
    frond_xy[float("-inf")] = ref_y

    # The first point from the set is added.
    p_x, p_y, p_z = sorted_ps[0]
    frond_xy[p_x] = p_y
    last_z = p_z  # the highest minimial point seen so far

    area = (ref_x - p_x) * (ref_y - p_y)
    volume = 0.0

    # And then all the other points are processed.
    for p_x, p_y, p_z in sorted_ps[1:, :]:
        # find greatest q_x, such that q_x <= p_x
        node_q = frond_xy.lower_bound_by_key(p_x)
        _, q_y = node_q

        if not (p_y < q_y):  # p is by dominated q
            continue

        volume += area * (p_z - last_z)
        last_z = p_z

        # remove dominated points and their area contributions
        prev_x = p_x
        prev_y = q_y
        node_s = frond_xy.succ(node_q)
        s_x, s_y = node_s

        while True:
            area -= (s_x - prev_x) * (ref_y - prev_y)
            if p_y > s_y:  # guaranteed by the sentinel point
                break
            prev_x = s_x
            prev_y = s_y
            dominated_node = node_s
            node_s = frond_xy.succ(node_s)
            s_x, s_y = node_s
            frond_xy.remove(dominated_node)

        # add the new point (here 's' is 't' in the paper)
        area += (s_x - p_x) * (ref_y - p_y)
        frond_xy[p_x] = p_y

    # Add last point's contribution to the volume
    volume += area * (ref_z - last_z)

    return volume


# TODO: Move these 3 classes/types to another file

# A custum Numpy datatype for structured access to coordinate data.
point3d_dt = np.dtype([("x", float), ("y", float), ("z", float)], align=True)


class SweepItem:
    """Models an item of the sweep structure."""

    def __init__(self, val: np.ndarray, idx: int) -> None:
        """Initialize the sweep item."""
        self.val = val
        self.idx = idx

    def __repr__(self) -> str:
        """Create a string representation of the sweep item."""
        return "SweepItem({}, {})".format(self.val, self.idx)


class Box3D:
    """Models and axis-parallel box in 3D."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        """Initialize the 3D box defined by two corner points."""
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:
        """Create string representation of the box."""
        return "Box3D[{}, {}]".format(self.lower, self.upper)

    def volume(self) -> float:
        """Compute the volume of the box."""
        return (
            (self.upper["x"] - self.lower["x"])
            * (self.upper["y"] - self.lower["y"])
            * (self.upper["z"] - self.lower["z"])
        )


def contributions_3d(
    ps: np.ndarray, ref_p: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of points.

    Parameters
    ----------
    ps
        The set of mutually non-dominated points.
    ref_p
        The reference point.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the EF algorithm (see p. 22 of :cite:`2020:hypervolume`) \
    presented by :cite:`2011-hypervolume-3d` for computing AllContributions. \
    The implementation differs from the presentation of the reference paper \
    in that it assumes a minimization problem (instead of maximization). \
    We also incorporate some aspects taken from :cite:`2008:shark`.
    """
    n = len(ps)

    if n == 0:
        return np.empty()

    if ref_p is None:
        zero = np.array([(0, 0, 0)], dtype=point3d_dt)[0]
        _ref_p = zero
    else:
        _ref_p = ref_p

    # Sort the points by their z-coordinate in ascending order
    sorted_idx = np.argsort(ps[:, 2])
    sorted_ps = ps[sorted_idx]

    # "Cast" to allow access by "key"
    ps = sorted_ps.view(dtype=point3d_dt).reshape(-1)
    _ref_p = _ref_p.view(dtype=point3d_dt)[0]

    # Initialization
    front_xy = RBTree()

    # Add boundary sentinels to the sweeping structure
    sentinel_x = np.array(
        [(_ref_p["x"], float("-inf"), float("-inf"))], dtype=point3d_dt
    )[0]
    front_xy[_ref_p["x"]] = SweepItem(sentinel_x, n)
    sentinel_y = np.array(
        [(float("-inf"), _ref_p["y"], float("-inf"))], dtype=point3d_dt
    )[0]
    front_xy[float("-inf")] = SweepItem(sentinel_y, n)

    # Create box lists and contribution vector
    boxes: List[Deque[Box3D]] = [deque([]) for _ in range(n + 1)]
    contribution = np.zeros(n + 1, dtype=float)
    contribution[n] = float("nan")

    # Process first point
    p0 = ps[0]
    upper0 = _ref_p.copy()
    upper0["z"] = float("nan")
    boxes[0].appendleft(Box3D(p0.copy(), upper0))
    front_xy[p0["x"]] = SweepItem(p0, 0)

    # Create pending points list
    pending = [SweepItem(p, i) for i, p in enumerate(ps[1:], 1)]

    # Process the rest of the points
    for p in pending:

        # find greatest q_x, such that q_x <= p_x
        node_q = front_xy.lower_bound_by_key(p.val["x"])
        left = node_q.item

        if not (p.val["y"] < left.val["y"]):  # p is dominated by q
            continue

        # (a) Find all points dominated by p, remove them
        # from the sweeping structure and add them to
        # the processing list 'dominated'.
        node_s = front_xy.succ(node_q)
        right = node_s.item
        dominated = []
        while True:
            if p.val["y"] > right.val["y"]:
                break
            dominated.append(right)
            node_d = node_s
            node_s = front_xy.succ(node_s)
            right = node_s.item
            front_xy.remove(node_d)
        front_xy[p.val["x"]] = p

        # (b) Process "left" region
        # Notice the symmetry with the paper's algorithm
        vol = 0.0
        while any(boxes[left.idx]):
            b = boxes[left.idx][-1]
            if p.val["x"] < b.lower["x"]:
                # This box is dominated at this z-level
                # so it can completed and added to the
                # volume contribution of the left neighbour.
                b.upper["z"] = p.val["z"]
                vol += b.volume()
                boxes[left.idx].pop()
            else:
                if p.val["x"] < b.upper["x"]:
                    # Stop removing boxes.
                    b.upper["z"] = p.val["z"]
                    vol += b.volume()
                    # Modify box to reflect the dominance
                    # of the left neighbour in this part
                    # of the L region.
                    b.upper["x"] = p.val["x"]
                    b.upper["z"] = float("nan")
                    b.lower["z"] = p.val["z"]
                    break
        contribution[left.idx] += vol

        # (c) Process the dominated points
        right_x = right.val["x"]
        for d in reversed(dominated):
            while any(boxes[d.idx]):
                b = boxes[d.idx].pop()
                b.upper["z"] = p.val["z"]
                contribution[d.idx] += b.volume()
            # Create new box
            upper = d.val.copy()
            upper["x"] = right_x
            upper["z"] = float("nan")
            lower = p.val.copy()
            lower["x"] = d.val["x"]
            b = Box3D(lower, upper)
            boxes[p.idx].appendleft(b)
            right_x = d.val["x"]

        upper = left.val.copy()
        upper["x"] = right_x
        upper["z"] = float("nan")
        lower = p.val.copy()
        b = Box3D(lower, upper)
        boxes[p.idx].appendleft(b)

        # (d) Process "right" region
        vol = 0.0
        right_x = right.val["x"]
        while any(boxes[right.idx]):
            b = boxes[right.idx][0]
            if b.upper["y"] >= p.val["y"]:
                b.upper["z"] = p.val["z"]
                vol += b.volume()
                right_x = b.upper["x"]
                boxes[right.idx].popleft()
            else:
                break

        if right_x > right.val["x"]:
            upper = p.val.copy()
            upper["x"] = right_x
            upper["z"] = float("nan")
            lower = right.val.copy()
            lower["z"] = p.val["z"]
            b = Box3D(lower, upper)
            boxes[right.idx].appendleft(b)
        contribution[right.idx] += vol

    # The paper uses a 'z sentinel' to close the remaining boxes.
    # Here we do it as in Shark's approach.
    for p in front_xy:
        while any(boxes[p.idx]):
            b = boxes[p.idx].pop()
            b.upper["z"] = _ref_p["z"]
            contribution[p.idx] += b.volume()

    reverse_idx = np.argsort(sorted_idx)
    return contribution[:-1][reverse_idx]


def contributions_3d_naive(
    ps: np.ndarray, ref_p: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of points.

    Parameters
    ----------
    ps
        The set of mutually non-dominated points.
    ref_p
        The reference point.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the brute-force approach to computing the hypervolume \
    contributions. Only for testing the other implementation, as done in \
    :cite:`2008:shark`.
    """
    if len(ps) == 0:
        return np.empty()

    if ref_p is None:
        ref_p = np.zeros_like(ps[0])

    contribution = np.zeros(len(ps))

    vol = calculate_3d(ps, ref_p)
    for i in range(len(ps)):
        qs = np.delete(ps, i, 0)
        contribution[i] = vol - calculate_3d(qs, ref_p)
    return contribution
