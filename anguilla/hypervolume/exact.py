"""Prototype implementations of different hypervolume algorithms."""

import numpy as np
from collections import deque
from typing import Deque, List, Optional
from anguilla.ds.rbtree import RBTree
from anguilla.dominance import non_dominated_sort

__all__ = ["hv2d", "hv3d", "hvkd", "hvc2d", "hvc3d"]

# A custom Numpy datatype for structured access to 3-D coordinate data.
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
    """Models an axis-parallel box in 3D."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        """Initialize the 3D box, defined by two corner points."""
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


def hv2d(points: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """Compute the exact hypervolume indicator for a set of 2-D points.

    Parameters
    ----------
    points
        The point set of mutually non-dominated points.
    reference: optional
        The reference point. \
        Otherwise assumed to be the component-wise maximum.

    Returns
    -------
    float
        The hypervolume indicator.

    Notes
    -----
    Ported from :cite:`2008:shark`.
    """
    if len(points) == 0:
        return 0.0

    if reference is None:
        ref_x, ref_y = np.max(points, axis=0)
    else:
        ref_x, ref_y = reference

    # Copy point set and sort along the x-axis in ascending order.
    sorted_idx = np.argsort(points[:, 0])
    sorted_ps = points[sorted_idx]

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


def hvc2d(points: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """Compute the exact hypervolume contributions for a set of 2-D points.

    Parameters
    ----------
    points
        The point set of mutually non-dominated points.
    reference: optional
        The reference point. \
        Otherwise assumed to be the component-wise maximum.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the algorithm described in Lemma 1 of :cite:`2007:mo-cma-es`.
    """
    if len(points) == 0:
        return 0.0

    if reference is None:
        ref_x, ref_y = np.max(points, axis=0)
    else:
        ref_x, ref_y = reference

    # Copy point set and sort along the x-axis in ascending order.
    sorted_idx = np.argsort(points[:, 0])
    sorted_ps = points[sorted_idx]

    contribution = np.zeros(len(points))

    contribution[0] = (sorted_ps[1][0] - sorted_ps[0][0]) * (
        reference[1] - sorted_ps[0][1]
    )
    contribution[-1] = (reference[0] - sorted_ps[-1][0]) * (
        sorted_ps[-2][1] - sorted_ps[-1][1]
    )

    for i in range(1, len(points) - 1):
        contribution[i] = (sorted_ps[i + 1][0] - sorted_ps[i][0]) * (
            sorted_ps[i - 1][1] - sorted_ps[i][1]
        )

    reverse_idx = np.argsort(sorted_idx)
    return contribution[reverse_idx]


def hv3d(points: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """Calculate the exact hypervolume indicator for a set of 3-D points.

    Parameters
    ----------
        points
            The point set of mutually non-dominated points.
        reference: optional
            The reference point. \
            Otherwise assumed to be the component-wise maximum.

    Returns
    -------
    float
        The hypervolume indicator.

    Notes
    -----
    Implements Algorithm 1 from :cite:`2009-hypervolume-hv3d`, \
    but with the difference of assuming a minimization problem. \
    See also Algorithm HV3D presented in :cite:`2020:hypervolume`, and the \
    explanation from sec. 4.1 of p. 6. of :cite:`2009-hypervolume-hv3d`.

    The differences in the implementation, w.r.t. the algorithm description \
    in the paper, are primarily due to them assuming a maximization \
    problem and the implementation the opposite (minimization).

    The following figure shows an example of how the algorithm is \
    transformed for working with a minimization problem. Note that \
    in both cases, the x-coordinates are assumed to be sorted in \
    ascending order.

    .. image:: /figures/hv3d_min.png
       :width: 750
       :alt: Example of the algorithm for minimization problem.
    """
    if len(points) == 0:
        return 0.0

    if reference is None:
        ref_x, ref_y, ref_z = np.max(points, axis=0)
    else:
        ref_x, ref_y, ref_z = reference

    # Sort the points by their z-coordinate in ascending order.
    sorted_idx = np.argsort(points[:, 2])
    sorted_ps = points[sorted_idx]

    # The algorithm works by performing sweeping in the z-axis,
    # and it uses a tree with balanced height as its sweeping structure
    # (e.g. an AVL tree) in which the keys are the x-coordinates and the
    # values are the y-coordinates.
    # Currently we use a red-black tree (as Shark uses std::map).
    # As in p. 6 of [2009:hypervolume-hv3d], the structure mantains the
    # x-coordinates in ascending order.
    front_xy = RBTree()

    # As explained in [2009:hypervolume-hv3d], we use two sentinel points
    # to ease the handling of boundary cases by ensuring that succ(p_x)
    # and pred(p_x) are defined for any other p_x in the tree.
    front_xy[ref_x] = float("-inf")
    front_xy[float("-inf")] = ref_y

    # The first point from the set is added.
    p_x, p_y, p_z = sorted_ps[0]
    front_xy[p_x] = p_y
    last_z = p_z  # the highest minimial point seen so far

    area = (ref_x - p_x) * (ref_y - p_y)
    volume = 0.0

    # And then all the other points are processed.
    for p_x, p_y, p_z in sorted_ps[1:, :]:
        # find greatest q_x, such that q_x <= p_x
        node_q = front_xy.lower_bound_by_key(p_x)
        q_y = node_q.item

        if not (p_y < q_y):  # p is by dominated q
            continue

        volume += area * (p_z - last_z)
        last_z = p_z

        # remove dominated points and their area contributions
        prev_x = p_x
        prev_y = q_y
        node_s = front_xy.succ(node_q)
        s_x = node_s.key
        s_y = node_s.item

        while True:
            area -= (s_x - prev_x) * (ref_y - prev_y)
            if p_y > s_y:  # guaranteed by the sentinel point
                break
            prev_x = s_x
            prev_y = s_y
            dominated_node = node_s
            node_s = front_xy.succ(node_s)
            s_x = node_s.key
            s_y = node_s.item
            front_xy.remove(dominated_node)

        # add the new point (here 's' is 't' in the paper)
        area += (s_x - p_x) * (ref_y - p_y)
        front_xy[p_x] = p_y

    # Add last point's contribution to the volume
    volume += area * (ref_z - last_z)

    return volume


def hvc3d(
    points: np.ndarray, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the hypervolume contribution for a set of 3-D points.

    Parameters
    ----------
    points
        The set of mutually non-dominated points.
    reference: optional
        The reference point. Otherwise assumed to be the component-wise \
        maximum.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the HVC3D algorithm (see p. 22 of :cite:`2020:hypervolume`) \
    presented by :cite:`2011:hypervolume-3d` for computing AllContributions. \
    The implementation differs from the presentation of the reference paper \
    in that it assumes a minimization problem (instead of maximization). \
    It also incorporates some implementation details taken from \
    :cite:`2008:shark`.
    """
    n = len(points)

    if n == 0:
        return np.empty()

    if reference is None:
        _ref_p = np.max(points, axis=0)
    else:
        _ref_p = reference

    # Sort the points by their z-coordinate in ascending order
    sorted_idx = np.argsort(points[:, 2])
    sorted_ps = points[sorted_idx]

    # "Cast" to allow access by "key"
    points = sorted_ps.view(dtype=point3d_dt).reshape(-1)
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
    p0 = points[0]
    upper0 = _ref_p.copy()
    upper0["z"] = float("nan")
    boxes[0].appendleft(Box3D(p0.copy(), upper0))
    front_xy[p0["x"]] = SweepItem(p0, 0)

    # Create pending points list
    pending = [SweepItem(p, i) for i, p in enumerate(points[1:], 1)]

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
                else:
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
            if b.upper["y"] > p.val["y"]:
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


def hvkd(
    points: np.ndarray, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the hypervolume indicator for a set of k-D points.

    Parameters
    ----------
    points
        The set of mutually non-dominated points.
    reference: optional
        The reference point. Otherwise assumed to be the component-wise \
        maximum.

    Returns
    -------
    float
        The hypervolume indicator.

    Notes
    -----
    Implements the basic version of the WFG algorithm \
    presented by :cite:`2012:hypervolume_wfg`. Incorporates aspects from \
    the implementation by :cite:`2008:shark` (URL: https://git.io/JtaJK). \
    It assumes minimization.
    """

    def limit_set(points, point):
        out = np.maximum(points, point)
        return out

    def box_volume(point, reference):  # 'incluhv' in the WFG paper
        return np.prod(reference - point)

    def wfg(points, reference):
        n = len(points)

        # This base case is from Shark:
        if n == 1:
            return box_volume(points[0], reference)

        # WFG recursive calls here:
        vol = 0.0
        for i in range(n):  # 'excluhv' in the WFG paper
            lset = limit_set(points[i + 1 :], points[i])
            ranks, _ = non_dominated_sort(lset, 1)
            ndset = lset[ranks == 1]
            vol += box_volume(points[i], reference) - wfg(ndset, reference)
        return vol

    n = len(points)
    if n == 0:
        return np.empty()

    if reference is None:
        reference = np.max(points, axis=0)

    sorted_idx = np.argsort(points[:, -1])  # sort by last component
    return wfg(points[sorted_idx], reference)
