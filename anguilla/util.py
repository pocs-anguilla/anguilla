"""This module contains utilitary/auxilliary functions."""
import math
import functools

from typing import Any, Optional, Callable, Tuple


def _exp_norm_chi_implementation(
    f: Callable[[float], float]
) -> Callable[[float], float]:
    """Decorate exp_norm_chi so that it is possible to vary its \
        implementation while keeping a single copy of the docstring."""
    try:
        import scipy.special

        @functools.wraps(f)
        def implementation(k: float) -> float:
            tmp = k * 0.5
            return (
                math.sqrt(2.0)
                * scipy.special.gamma(tmp + 0.5)
                / scipy.special.gamma(tmp)
            )

    except ImportError:

        @functools.wraps(f)
        def implementation(k: float) -> float:
            return math.sqrt(2.0) * (
                1.0 - 1.0 / (4.0 * k) + 1.0 / (21.0 * k * k)
            )

    return implementation


@_exp_norm_chi_implementation
def exp_norm_chi(k: float) -> float:
    """Approximate the expectation of a random variable defined as \
        the 2-norm of another sampled from a k-dimensional \
            multivariate Gaussian distribution.

    Parameters
    ----------
    k
        The dimensionality of the multivariate Gaussian distribution.

    Returns
    -------
        An approximation of said expectation.

    Notes
    -----
    The formula is presented in p. 28 of :cite:`2016:cma-es-tutorial`:

    .. math::
        \\mathbb{E} [ ||Z|| ] = \\sqrt{2} \
        \\Gamma \\left( \\frac{k+1}{2} \\right) \
        / \\Gamma \\left( \\frac{k}{2} \\right) \\approx \\sqrt{k} \
            \\left(1 + \\frac{k+1}{2} + \\frac{1}{21 k^2} \\right)

    where

    .. math::
        Z \\sim N(0, I_k)

    If Scipy is available, use its gamma function implementation to \
        compute the approximation.
    """
    pass


# TODO: transform RB-tree into TRB-tree.
class RBTree:
    """A Red-Black Tree.

    Notes
    -----
    Implements a Red-Black tree as described in Chapter 13 of \
    :cite:`2009:clrs`.
    """

    class _Node:
        def __init__(self, key: Any, value: Any) -> None:
            self.key = key
            self.value = value
            self.is_red = False
            self.left = self
            self.right = self
            self.parent = self

        def __repr__(self) -> str:
            return "({}, {})".format(self.key, self.value)

    def __init__(self) -> None:
        """Initialize the tree."""
        self._nil = RBTree._Node(None, None)  # sentinel
        self._root = self._nil

    def __repr__(self) -> str:
        def _get_nodes(x):
            if x is not self._nil:
                return _get_nodes(x.left) + [x] + _get_nodes(x.right)
            else:
                return []

        return "RBTree{}".format(_get_nodes(self._root))

    def _create_node(self, key: Any, value: Any) -> _Node:
        x = RBTree._Node(key, value)
        x.left = self._nil
        x.right = self._nil
        x.parent = self._nil
        return x

    def _left_rotate(self, x: _Node) -> None:
        y = x.right
        x.right = y.left
        if y.left is not self._nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x: _Node) -> None:
        y = x.left
        x.left = y.right
        if y.right is not self._nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _fix_insert(self, z: _Node) -> None:
        while z.parent.is_red:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.is_red:
                    z.parent.is_red = False
                    y.is_red = False
                    z.parent.parent.is_red = True
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.is_red = False
                    z.parent.parent.is_red = True
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.is_red:
                    z.parent.is_red = False
                    y.is_red = False
                    z.parent.parent.is_red = True
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.is_red = False
                    z.parent.parent.is_red = True
                    self._left_rotate(z.parent.parent)
        self._root.is_red = False

    def _transplant(self, u: _Node, v: _Node) -> None:
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _fix_remove(self, x: _Node):
        while x is not self._root and not x.is_red:
            if x is x.parent.left:
                w = x.parent.right
                if w.is_red:
                    w.is_red = False
                    x.parent.is_red = True
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if not w.left.is_red and not w.right.is_red:
                    w.is_red = True
                    x = x.parent
                else:
                    if not w.right.is_red:
                        w.left.is_red = False
                        w.is_red = True
                        self._right_rotate(w)
                        w = x.parent.right
                    w.is_red = x.parent.is_red
                    x.parent.is_red = False
                    w.right.is_red = False
                    self._left_rotate(x.parent)
                    x = self._root
            else:
                w = x.parent.left
                if w.is_red:
                    w.is_red = False
                    x.parent.is_red = True
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if not w.right.is_red and not w.left.is_red:
                    w.is_red = True
                    x = x.parent
                else:
                    if not w.left.is_red:
                        w.right.is_red = False
                        w.is_red = True
                        self._left_rotate(w)
                        w = x.parent.left
                    w.is_red = x.parent.is_red
                    x.parent.is_red = False
                    w.left.is_red = False
                    self._right_rotate(x.parent)
                    x = self._root
        x.is_red = False

    def _find(self, key: Any) -> Optional[_Node]:
        x = self._root
        while x is not self._nil:
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x
        return None

    def is_empty(self) -> bool:
        """Determine if the tree is empty."""
        return self._root is self._nil

    def succ(self, key: Any) -> Tuple[Any, Any]:
        """Return the succesor for the given key."""
        x = self._find(key)
        if x is not None:
            if x.right is not self._nil:
                x = x.right  # x = min(x.right)
                while x.left is not self._nil:
                    x = x.left
                return x.key, x.value
            y = x.parent
            while y is not self._nil and x is y.right:
                x = y
                y = y.parent
            return y.key, y.value
        else:
            raise KeyError(str(key))

    def pred(self, key: Any) -> Tuple[Any, Any]:
        """Return the predecesor for the given key."""
        x = self._find(key)
        if x is not None:
            if x.left is not self._nil:
                x = x.left  # x = max(x.left)
                while x.right is not self._nil:
                    x = x.right
                return x.key, x.value
            y = x.parent
            while y is not self._nil and x is y.left:
                x = y
                y = y.parent
            return y.key, y.value
        else:
            raise KeyError(str(key))

    def lower_bound(self, key: Any) -> Tuple[Any, Any]:
        """Return the lower bound for the given key. \
        That is, the greatest that is less than or equal to the key.
        
        Notes
        -----
        The given key may not exist in the tree.

        
        Returns
        -------
        Any, Any
            Returns the key and value of the lower bound if it exists.
            Otherwise returns None and None.
            
        """
        y = self._nil
        x = self._root
        while x is not self._nil:
            y = x
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x.key, x.value
        if y is self._nil:
            return None, None
        elif key < y.key:
            return self.pred(y.key)
            # start inlining of pred
            x = y
            if x.left is not self._nil:
                x = x.left  # x = max(x.left)
                while x.right is not self._nil:
                    x = x.right
                return x.key, x.value
            y = x.parent
            while y is not self._nil and x is y.left:
                x = y
                y = y.parent
            return y.key, y.value
            # end inlining of pred
        else:
            return y.key, y.value

    def insert(self, key: Any, value: Any) -> None:
        """Insert the given value for the given key."""
        z = self._create_node(key, value)
        y = self._nil
        x = self._root
        while x is not self._nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:  # TODO: support dictionary interface instead?
                x = x.right
        z.parent = y
        if y is self._nil:
            self._root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.is_red = True
        self._fix_insert(z)

    def remove(self, key: Any) -> None:
        """Remove the given key."""
        z = self._find(key)
        if z is not None:
            y = z
            y_was_red = y.is_red
            if z.left is self._nil:
                x = z.right
                self._transplant(z, z.right)
            elif z.right is self._nil:
                x = z.left
                self._transplant(z, z.left)
            else:
                y = z.right  # y = min(z.right)
                while y.left is not self._nil:
                    y = y.left

                y_was_red = y.is_red
                x = y.right
                if y.parent is z:
                    x.parent = y
                else:
                    self._transplant(y, y.right)
                    y.right = z.right
                    y.right.parent = y

                self._transplant(z, y)
                y.left = z.left
                y.left.parent = y
                y.is_red = z.is_red
            if not y_was_red:
                self._fix_remove(x)
        else:
            raise KeyError(str(key))
