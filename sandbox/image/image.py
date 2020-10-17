from __future__ import annotations  # type: ignore

import imageio      # type: ignore
import numpy as np  # type: ignore
import PIL.Image    # type: ignore

from . import Point
from itertools import product
from typing import Iterator, Tuple


class Image:
    """
    An Image object represents an image.  Simple enough, right?
    """

    def __init__(self, path: str = None) -> None:
        """
        Constructs an Image with the given path.  Setting the path to None means
        that this Image should not be loaded from (or saved to) disk.

        Args:
            path: Path to the Image.
        """
        self.__path = path
        self.__pixels: np.ndarray = None
        self.__filled: np.ndarray = None

    @property
    def area(self) -> int:
        """Returns the number of Pixels in this Image."""
        return self.height * self.width

    @property
    def center(self) -> Point:
        """Returns the center Point of this Image."""
        return Point(self.width // 2, self.height // 2)

    @property
    def height(self) -> int:
        """Returns the height of this Image."""
        return self.__pixels.shape[0]

    @property
    def shape(self) -> np.ndarray:
        """Returns the shape of this Image (height, width, channels)."""
        return self.__pixels.shape

    @property
    def size(self) -> np.ndarray:
        """Returns the size of this Image (height, width)."""
        return (self.height, self.width)

    @property
    def width(self) -> int:
        """Returns the width of this Image."""
        return self.__pixels.shape[1]

    def clamp(self, point: Point) -> Point:
        """Returns the closest Point to the given Point in this Image."""
        w, h = self.width, self.height
        x = min(w - 1, max(0, point.x))
        y = min(h - 1, max(0, point.y))
        return Point(x, y)

    def copy(self) -> Image:
        """Returns a deep copy of this Image."""
        image = Image(self.__path)
        image.create(self.width, self.height)
        image.__pixels = np.copy(self.__pixels)
        image.__filled = np.copy(self.__filled)
        return image

    def corners(self) -> Tuple[Point, Point]:
        """Returns the minimum and maximum Points in this Image."""
        return Point(0, 0), Point(self.width - 1, self.height - 1)

    def cover(self, order: str = 'raster') -> Iterator[Point]:
        """
        Yields each Point in this Image in the specified order.

        Args:
            order: The order in which to yield the Points (i.e., 'raster', 'spiral').
        
        Returns:
            The Points in this Image in the specified order.
        """
        if order == 'raster':
            for y, x in product(range(self.height), range(self.width)):
                yield Point(x, y)
        elif order == 'spiral':
            # First, yield the center Point.
            yield self.center
            # Then, yield Points in the progression of the following spiral:
            #     +-----------+---+
            #     | 4 | <------ 3 |
            #     | | +-------+---+
            #     | | |       | ^ |
            #     | v |       | | |
            #     +---+-------+ | |
            #     | 1 ------> | 2 |
            #     +---+-----------+
            layers = max(self.center.x + 1, self.center.y + 1, self.width - self.center.x, self.height - self.center.y)
            for layer in range(1, layers):
                min_corner = Point(self.center.x - layer, self.center.y - layer)
                max_corner = Point(self.center.x + layer, self.center.y + layer)
                # Iterate along the bottom of the spiral.
                for x in range(min_corner.x, max_corner.x):
                    point = Point(x, max_corner.y)
                    if point in self:
                        yield point
                # Iterate along the right of the spiral.
                for y in range(max_corner.y, min_corner.y, -1):
                    point = Point(max_corner.x, y)
                    if point in self:
                        yield point
                # Iterate along the top of the spiral.
                for x in range(max_corner.x, min_corner.x, -1):
                    point = Point(x, min_corner.y)
                    if point in self:
                        yield point
                # Iterate along the left of the spiral.
                for y in range(min_corner.y, max_corner.y):
                    point = Point(min_corner.x, y)
                    if point in self:
                        yield point
        else:
            raise NotImplementedError(f'Order "{order}" is not supported.')

    def create(self, width: int, height: int) -> None:
        """Initializes a black RGB Image with the given size."""
        assert width > 0 and height > 0, "Size cannot be zero or negative."
        self.__pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self.__filled = np.full((height, width), False)

    def extract(self, center: Point, padding: int, mode: str = 'drop') -> Image:
        """
        Extracts a window from this Image centred at the given Point with the
        specified padding.

        Args:
            center: The center of the window.
            padding: The padding of the window.
            mode: Approach to handling pixels that fall outside this Image.
                  Supported modes include 'drop', 'reflect', and 'wrap'.

        Returns:
            Image object with the specified window.
        """
        # Create a window using the padding dimensions.
        window = Image()
        window.create(2 * padding + 1, 2 * padding + 1)
        if mode == 'drop':
            # Find the corners of the window in this Image.
            min_corner = self.clamp(Point(center.x - padding, center.y - padding))
            max_corner = self.clamp(Point(center.x + padding, center.y + padding))
            # Find the corner in the window which corresponds to the minimum corner in this Image.
            offset = Point(padding - (center.x - min_corner.x), padding - (center.y - min_corner.y))
            # Populate the window pixels and filled arrays.
            w, h = max_corner.x - min_corner.x + 1, max_corner.y - min_corner.y + 1
            window.__pixels[offset.y:offset.y + h, offset.x:offset.x + w] = self.__pixels[min_corner.y:max_corner.y + 1, min_corner.x:max_corner.x + 1]
            window.__filled[offset.y:offset.y + h, offset.x:offset.x + w] = self.__filled[min_corner.y:max_corner.y + 1, min_corner.x:max_corner.x + 1]
            return window
        elif mode == 'reflect' or mode == 'wrap':
            padded_pixels = np.stack([np.pad(self.__pixels[:, :, c], padding, mode=mode) for c in range(3)], axis=2)
            padded_filled = np.pad(self.__filled, padding, mode=mode)
            # Adjust the window offset to accommodate the padding.
            offset = center + Point(padding, padding)
            # Populate the window pixels and filled arrays.
            window.__pixels = padded_pixels[offset.y - padding:offset.y + padding + 1, offset.x - padding:offset.x + padding + 1]
            window.__filled = padded_filled[offset.y - padding:offset.y + padding + 1, offset.x - padding:offset.x + padding + 1]
            return window
        else:
            raise NotImplementedError(f'Mode "{mode}" is not supported.')

    def filled(self, center: Point, padding: int, mode: str = 'drop', causal: bool = False) -> np.ndarray:
        """
        Reports which pixels in the window centred at the given Point with the
        specified padding have been rendered in this Image.

        Args:
            center: The center of the window.
            padding: The padding of the window.
            mode: Approach to handling pixels that fall outside this Image.
                  Supported modes include 'drop', 'reflect', and 'wrap'.
            causal: Restrict the set of Points that can be True to those which
                    lie above or the left of the given Point.

        Returns:
            A two-dimensional array with a False entry in every position of the
            window which has yet to be rendered.
        """
        filled = self.extract(center, padding, mode).__filled
        if causal:
            filled = filled & (np.arange(filled.size).reshape(filled.shape) <= filled.size // 2)
        return filled

    def load(self) -> None:
        """Loads this Image from disk."""
        assert self.__path, "Path cannot be empty or set to None."
        self.__pixels = imageio.imread(self.__path)
        self.__filled = np.full(self.__pixels.shape[:2], True)

    def paste(self, center: Point, window: Image) -> None:
        """
        Pastes the given window into this Image at a position which is centred
        about the specified Point.

        Args:
            center: The center position of the window in this Image.
            window: The window to paste.
        """
        padding = window.width // 2
        min_corner = self.clamp(Point(center.x - padding, center.y - padding))
        max_corner = self.clamp(Point(center.x + padding, center.y + padding))
        # Find the corner in the window which corresponds to the minimum corner in this Image.
        offset = Point(padding - (center.x - min_corner.x), padding - (center.y - min_corner.y))
        # Populate the window pixels and filled arrays.
        w, h = max_corner.x - min_corner.x + 1, max_corner.y - min_corner.y + 1
        self.__pixels[min_corner.y:max_corner.y + 1, min_corner.x:max_corner.x + 1] = window.__pixels[offset.y:offset.y + h, offset.x:offset.x + w]
        self.__filled[min_corner.y:max_corner.y + 1, min_corner.x:max_corner.x + 1] = window.__filled[offset.y:offset.y + h, offset.x:offset.x + w]

    def project(self, point: Point, mode: str = 'wrap') -> Point:
        """Projects the given Point to this Image."""
        if mode == 'wrap':
            # Example: 0 1 2 3 4 <-- 0 1 2 3 4 --> 0 1 2 3 4
            x = point.x % self.width
            y = point.y % self.height
        elif mode == 'reflect':
            # Example: 0 1 2 3 4 3 2 1 <-- 0 1 2 3 4 --> 3 2 1 0 1 2 3 4
            def reflect(value, boundary):
                """Reflects the given value across the specified boundary"""
                period = 2 * boundary - 2
                modulus = value % period
                return min(modulus, period - modulus)
            x = reflect(point.x, self.width)
            y = reflect(point.y, self.height)
        else:
            raise NotImplementedError(f'Mode "{mode}" is not supported.')
        return Point(x, y)

    def resize(self, width: int, height: int) -> None:
        """Resizes this Image to the given width and height."""
        assert width > 0 and height > 0, "Size cannot be zero or negative."
        size = (width, height)
        self.__pixels = np.array(PIL.Image.fromarray(self.__pixels).resize(size))
        self.__filled = np.full((height, width), True)

    def save(self) -> None:
        """Saves this Image to disk."""
        assert self.__path, "Path cannot be empty or set to None."
        imageio.imsave(self.__path, self.__pixels)

    def __eq__(self, other: object) -> bool:
        """Returns True if this Image has the same pixels as the given Image."""
        if not isinstance(other, Image):
            return NotImplemented
        return np.array_equal(self.__pixels, other.__pixels)

    def __contains__(self, point: Point) -> bool:
        """Reports whether the given Point is in this Image."""
        min_corner, max_corner = self.corners()
        return (min_corner.x <= point.x <= max_corner.x) and \
               (min_corner.y <= point.y <= max_corner.y)

    def __getitem__(self, key: object) -> np.ndarray:
        """Returns the colour of the given slice or Point in this Image."""
        if isinstance(key, tuple):
            return self.__pixels[key]
        elif isinstance(key, Point):
            return self.__pixels[key.y, key.x]
        else:
            raise NotImplementedError(f"Image.__getitem__() is not implemented for type {type(key)}.")

    def __setitem__(self, key: object, colour: np.ndarray) -> None:
        """Sets the given slice or Point in this Image to the specified colour."""
        if isinstance(key, tuple):
            self.__pixels[key] = colour
            self.__filled[key] = True
        elif isinstance(key, Point):
            self.__pixels[key.y, key.x] = colour
            self.__filled[key.y, key.x] = True
        else:
            raise NotImplementedError(f"Image.__setitem__() is not implemented for type {type(key)}.")
