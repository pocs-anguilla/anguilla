"""Implements a red-black tree."""

from typing import Any, Generator, Optional, Tuple, List


class RBNode:
    """Models the node of a red-black tree."""

    def __init__(self, key: Any, item: Any) -> None:
        """Initialize the node."""
        self.key = key
        self.item = item
        self.is_red = False
        self.left = self
        self.right = self
        self.parent = self

    def __repr__(self) -> str:
        """Create a string representation of the node."""
        return "({}, {})".format(self.key, self.item)


class RBTree:
    """Models a red-black tree.

    Notes
    -----
    The implementation uses parent pointers and follows closely \
    the presentation from Chapter 13 of :cite:`2009:clrs`.
    """

    def __init__(self) -> None:
        """Initialize the tree."""
        self._nil = RBNode(None, None)  # sentinel
        self._root = self._nil

    def __iter__(self) -> Generator[Any, Any, None]:
        """Return an iterator doing an in-order traversal."""

        def _iterator(x: Any) -> Any:
            if x is not self._nil:
                yield from _iterator(x.left)
                yield x.item
                yield from _iterator(x.right)

        return _iterator(self._root)

    def __repr__(self) -> str:
        """Return a string representation of the tree."""

        def _get_nodes(x: Any) -> List[Any]:
            if x is not self._nil:
                return _get_nodes(x.left) + [x] + _get_nodes(x.right)
            return []

        return "RBTree{}".format(_get_nodes(self._root))

    def __getitem__(self, key: Any) -> Any:
        """Get the item corresponding a the given key.

        Parameters
        ----------
        key
            The key of the item.

        Returns
        -------
        Any
            The item corresponding to the given key.

        Raises
        ------
        KeyError
            The given key is not in the tree.
        """
        x = self._find(key)
        if x is not None:
            return x.item
        raise KeyError(str(key))

    def _create_node(self, key: Any, value: Any) -> RBNode:
        """Create a new node for the item."""
        x = RBNode(key, value)
        x.left = self._nil
        x.right = self._nil
        x.parent = self._nil
        return x

    def _left_rotate(self, x: RBNode) -> None:
        """Left rotate the given subtree."""
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

    def _right_rotate(self, x: RBNode) -> None:
        """Right rotate the given subtree."""
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

    def _fix_insert(self, z: RBNode) -> None:
        """Perform the insertion fixes."""
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

    def _transplant(self, u: RBNode, v: RBNode) -> None:
        """Perform the node transplant (used by remove)."""
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _fix_remove(self, x: RBNode) -> None:
        """Perform the remove fixes."""
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

    def _find(self, key: Any) -> Optional[RBNode]:
        """Find the first node with the given key."""
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
        """Determine if the tree is empty.

        Returns
        bool
            The tree is empty.
        """
        return self._root is self._nil

    def is_nil(self, node: RBNode) -> bool:
        """Determine is a node is nil.

        Parameters
        ----------
        node
            The node to test.

        Returns
        bool
            The node is nil.
        """
        return self._nil is node

    def succ_by_key(self, key: Any) -> Tuple[Any, Any]:
        """Return the succesor for the given key.

        Parameters
        ----------
        key
            The key for which to look the successor.

        Raises
        ------
        KeyError
            The provided key is not in the tree.

        Returns
        -------
        Any, Any
            The key and value of the successor.
        """
        x = self._find(key)
        if x is not None:
            succ = self.succ(x)
            return succ.key, succ.item
        raise KeyError(str(key))

    def succ(self, x: RBNode) -> RBNode:
        """Return the successor for the node.

        Parameters
        ----------
        x
            The node for which to return the successor.

        Notes
        -----
        Undefined behaviour if the given node is not part of the tree.

        Returns
        -------
        RBNode
            The successor node.
        """
        if x.right is not self._nil:
            x = x.right  # find min of the right subtree
            while x.left is not self._nil:
                x = x.left
            return x
        y = x.parent
        while y is not self._nil and x is y.right:
            x = y
            y = y.parent
        return y

    def pred_by_key(self, key: Any) -> Tuple[Any, Any]:
        """Return the predecessor for the given key.

        Parameters
        ----------
        key
            The key for which to look the predecessor.

        Raises
        ------
        KeyError
            The provided key is not in the tree.

        Returns
        -------
        Any, Any
            The key and value of the predecessor.
        """
        x = self._find(key)
        if x is not None:
            pred = self.pred(x)
            return pred.key, pred.item
        raise KeyError(str(key))

    def pred(self, x: RBNode) -> RBNode:
        """Return the predecessor for the given node.

        Parameters
        ----------
        x
            The node for which to return the predecessor.

        Notes
        -----
        Undefined behaviour if the given node is not part of the tree.

        Returns
        -------
        RBNode
            The predecessor node, if it exists.
        """
        if x.left is not self._nil:
            x = x.left  # find max of left subtree
            while x.right is not self._nil:
                x = x.right
            return x
        y = x.parent
        while y is not self._nil and x is y.left:
            x = y
            y = y.parent
        return y

    def lower_bound_by_key(self, key: Any) -> RBNode:
        """Give a node representing the lower bound for the given key, \
        i.e., the node whose key is the greatest that is less than or \
        equal to the given key.

        Parameters
        ----------
        key
            The key for which to find the lower bound. It doesn't need to \
                exist in the tree.

        Notes
        -----
        Implementation adapted from that of https://git.io/JIIzK \
        (Mozman's bintree).

        Returns
        -------
        RBNode
            The node representing the lower bound, if it exists.
        """
        x = self._root
        result = self._nil
        while x is not self._nil:
            if key < x.key:
                x = x.left
            elif key > x.key:
                if (result is self._nil) or (x.key > result.key):
                    result = x
                x = x.right
            else:
                result = x
                break
        return result

    def __setitem__(self, key: Any, item: Any) -> None:
        """Insert (or update) the item corresponding to a given key.

        Parameters
        ----------
        key
            The key of the item.
        item
            The new item.
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
                x.item = item
                return
        z = self._create_node(key, item)
        z.parent = y
        if y is self._nil:
            self._root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.is_red = True
        self._fix_insert(z)

    def remove_by_key(self, key: Any) -> None:
        """Remove the given key.

        Parameters
        ----------
        key
            The key to remove.

        Raises
        ------
        KeyError
            The provided key doesn't exist in the tree.
        """
        z = self._find(key)
        if z is not None:
            self.remove(z)
        raise KeyError(str(key))

    def remove(self, z: RBNode) -> None:
        """Remove the given node.

        Parameters
        ----------
        z
            The node to remove.

        Notes
        -----
        Undefined behaviour if the given node is not part of the tree.
        """
        y = z
        y_was_red = y.is_red
        if z.left is self._nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self._nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = z.right  # find min of right subtree
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
