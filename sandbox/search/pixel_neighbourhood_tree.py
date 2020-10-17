from __future__ import annotations  # type: ignore

import numpy as np  # type: ignore
import random
import tqdm         # type: ignore

from collections import deque as Deque
from image import Image, Point
from scipy import signal  # type: ignore
from typing import Collection, Iterable, List, Sequence


class PixelNeighbourhoodTree:
    """
    A PixelNeighbourhoodTree object represents a TSVQ tree.  This data
    structure contains a single method, PixelNeighbourhoodTree.query(),
    which finds a pixel with a neighbourhood that is similar to the
    neighbourhood of a given pixel in (amortised) logarithmic time.

    The implementation of PixelNeighbourhoodTree is roughly based on the
    description of a TSVQ tree given in https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf.
    """

    def __init__(self, image: Image, branching_factor: int, neighbours: np.ndarray, description: str = None) -> None:
        """
        Constructs a new PixelNeighbourhoodTree from the given Image
        using the specified neighbourhood mask, branching factor, and
        progress bar description.

        Args:
            image: The Image.
            branching_factor: The maximum number of children a Node in
                              the TSVQ tree can have.
            neighbours: A mask indicating which offsets are considered to
                        be neighbours of the center offset.
            description: Description of the TQDM progress bar.  Setting
                         this parameter to None disables the progress bar.
        """
        assert branching_factor > 1, "The minimum branching factor is two."
        assert (len(neighbours.shape) == 2
                and neighbours.shape[0] == neighbours.shape[1]
                and neighbours.shape[0] % 2 == 1), "Neighbours array should be a square matrix with an odd side length."
        self.__image = image
        self.__neighbours = neighbours

        # Construct a 2D Gaussian kernel from the neighbours matrix.  The
        # tiling at the end is necessary to represent each colour channel.
        gaussian_1D = signal.gaussian(len(neighbours), std=len(neighbours) // 2)
        gaussian_2D = np.outer(gaussian_1D, gaussian_1D)
        self.__weights = np.tile(np.extract(neighbours, gaussian_2D), 3)

        # Precompute the neighbourhoods of each Point in the Image.
        self.__neighbourhoods = np.zeros(
            shape=(image.height, image.width, len(self.__weights)),
            dtype=np.float64
        )
        for point in image.cover():
            self.__neighbourhoods[point.y, point.x, :] = self.__extract_neighbourhood(image, point)

        # Initialize the root Node with all the Points in the Image.
        self.__root = Node(self.__average_neighbourhood(image.cover()), list(image.cover()))

        # Construct the TSVQ tree by maintaining a frontier of Nodes and
        # applying a variant of Lloyd's algorithm to decompose each Node
        # into several Nodes with smaller numbers of Points.
        frontier = Deque([self.__root])
        with tqdm.tqdm(desc=description, disable=description is None) as progress:
            while frontier:
                parent = frontier.popleft()
                progress.update(1)

                # Skip this Node if it is impossible to uniquely sample
                # the required number of Points.
                if len(parent.points) < branching_factor:
                    continue

                # Initialize the child Nodes at random Points from the
                # parent Node.
                initial_points = random.sample(parent.points, branching_factor)
                neighbourhoods = np.array([self.__neighbourhoods[point.y, point.x, :] for point in initial_points])

                # Iteratively partition the Points among the
                # neighbourhoods and update each neighbourhood to be the
                # centroid of its Points until convergence.
                prev_partition = np.zeros(len(parent.points), dtype=np.uint8)
                next_partition = self.__partition_points(neighbourhoods, parent.points)
                neighbourhoods = self.__erase_empty_neighbourhoods(neighbourhoods, next_partition)
                while np.any(prev_partition != next_partition):
                    # Update the neighbourhoods.
                    for i, neighbourhood in enumerate(neighbourhoods):
                        points = [parent.points[index] for index in np.flatnonzero(next_partition == i)]
                        neighbourhood[...] = self.__average_neighbourhood(points)
                    # Update the partition.
                    prev_partition = next_partition
                    next_partition = self.__partition_points(neighbourhoods, parent.points)
                    # Remove any neighbourhoods without any Points.
                    neighbourhoods = self.__erase_empty_neighbourhoods(neighbourhoods, next_partition)

                # There is no use expanding a Node that has only one child.
                if len(neighbourhoods) == 1:
                    continue

                # Transform the neighbourhoods into Nodes and add them to
                # both the frontier and the parent Node.
                for i, neighbourhood in enumerate(neighbourhoods):
                    points = [parent.points[index] for index in np.flatnonzero(next_partition == i)]
                    child = Node(neighbourhood, points)
                    parent.children.append(child)
                    frontier.append(child)

    def query(self, image: Image, point: Point, mode: str = 'wrap') -> Point:
        """
        Returns a Point from the Image used to construct this
        PixelNeighbourhoodTree which has a similar neighbourhood to the
        given Point in the specified Image.

        Args:
            image: The Image associated with the Point.
            point: The Point in the Image with the desired neighbourhood.
            mode: Approach to handling pixels outside the given Image.
        """
        point_vector = self.__extract_neighbourhood(image, point, mode)
        distance = lambda node: np.sum(self.__weights * (point_vector - node.vector)**2)
        node = self.__root
        while node.children:
            node = min(node.children, key=distance)
        return random.choice(node.points)

    def __erase_empty_neighbourhoods(self, neighbourhoods: np.ndarray, partition: np.ndarray) -> np.ndarray:
        """
        Erases each neighbourhood with no Points in the specified partition.

        Args:
            neighbourhoods: The neighbourhoods to check.
            partition: A partitioning of Points among the neighbourhoods.
                       This function will perform an in-place update of
                       the values in this partition to match the returned
                       neighbourhoods.

        Returns:
            The array of non-empty neighbourhoods.
        """
        missing_indices = np.setdiff1d(np.arange(len(neighbourhoods)), np.unique(partition))
        for index in np.sort(missing_indices)[::-1]:
            neighbourhoods = np.delete(neighbourhoods, index, axis=0)
            partition[np.argwhere(partition > index)] -= 1
        return neighbourhoods

    def __extract_neighbourhood(self, image: Image, point: Point, mode: str = 'reflect') -> np.ndarray:
        """
        Extracts the pixel neighbourhood around the given Point in the
        provided Image.

        Args:
            image: The Image.
            point: The Point at the center of the neighbourhood.
            mode: Approach to handling pixels outside the given Image.
        
        Returns:
            A vector representation of the specified pixel neighbourhood.
        """
        window = image.extract(point, len(self.__neighbours) // 2, mode)
        pixels = window[:, :].transpose(2, 0, 1)
        channels = [np.extract(self.__neighbours, channel) for channel in pixels]
        return np.concatenate(channels, axis=None)

    def __average_neighbourhood(self, points: Iterable[Point]) -> np.ndarray:
        """
        Computes the average neighbourhood of the given Points in the Image that
        was used to construct this PixelNeighbourhoodTree.

        Args:
            points: The Points whose neighbourhoods should be averaged.
        
        Returns:
            The average pixel neighourhoods of the given Points.
        """
        vectors = self.__neighbourhoods.reshape(self.__image.area, -1)
        indices = [point.y * self.__image.height + point.x for point in points]
        return np.sum(np.take(vectors, indices, axis=0), axis=0) / len(indices)
    
    def __partition_points(self, neighbourhoods: np.ndarray, points: Collection[Point]) -> np.ndarray:
        """
        Partitions the given Points among the provided neighbourhoods.

        Args:
            neighbourhoods: The array of neighbourhood assignment targets.
            points: The Points to be partitioned.

        Returns:
            An array where the element at each index indicates the neighbourhood
            assigned to that Point.
        """
        partition = np.zeros(len(points), dtype=np.uint8)
        for i, point in enumerate(points):
            reference = self.__neighbourhoods[point.y, point.x, :]
            distances = np.sum(self.__weights * (reference - neighbourhoods)**2, axis=1)
            partition[i] = np.argmin(distances)
        return partition


class Node:
    """
    A Node object is a POD type which represents a node in a TSVQ tree.

    Attributes:
        vector: The vector representing the neighbourhood of this Node.
        points: The Points belonging to this Node.
        children: The children of this Node.
    """

    def __init__(self, neighbourhood: np.ndarray, points: Sequence[Point]) -> None:
        """
        Constructs a new Node with the given neighbourhood and Points.

        Args:
            neighbourhood: The pixel neighbourhood of this Node.
            points: The Point cluster belonging to this Node.
        """
        assert len(neighbourhood.shape) == 1, "Neighbourhood must be a flat array."
        assert len(points) > 0, "Number of Points must be a strictly positive integer."
        self.vector = neighbourhood
        self.points = points
        self.children: List[Node] = []
