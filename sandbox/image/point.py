from __future__ import annotations  # type: ignore


class Point:
    """
    A Point object represents the location of a pixel in an Image.
    """

    def __init__(self, x: int, y: int) -> None:
        """
        Constructs a new Point with the given coordinates.

        Args:
            x: The x-coordinate of the Point.
            y: The y-coordinate of the Point.
        """
        self.__x = x
        self.__y = y

    @property
    def x(self) -> int:
        """Returns the x-coordinate of this Point."""
        return self.__x

    @property
    def y(self) -> int:
        """Returns the y-coordinate of this Point."""
        return self.__y

    def __hash__(self) -> int:
        """Returns a hash of this Point."""
        return hash((self.x, self.y))

    def __add__(self, other: Point) -> Point:
        """Returns the element-wise addition of this Point and the given Point."""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        """Returns the element-wise subtraction of this Point and the given Point."""
        return Point(self.x - other.x, self.y - other.y)

    def __rmul__(self, scalar: int) -> Point:
        """Returns the element-wise multiplication of this Point with the given scalar."""
        return Point(self.x * scalar, self.y * scalar)

    def __floordiv__(self, scalar: int) -> Point:
        """Returns the element-wise integer division of this Point by the given scalar."""
        return Point(self.x // scalar, self.y // scalar)

    def __eq__(self, other: object) -> bool:
        """Returns True if this Point has the same coordinates as the given Point."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: object) -> bool:
        """Returns True if this Point lies to the left or strictly below the given Point."""
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __repr__(self) -> str:
        """Returns a string representation of this Point."""
        return f'({self.x}, {self.y})'
