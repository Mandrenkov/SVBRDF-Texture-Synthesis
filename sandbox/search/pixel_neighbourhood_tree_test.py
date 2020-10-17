import numpy as np  # type: ignore
import unittest

from . import PixelNeighbourhoodTree
from image import Image, Point


class TestPixelNeighbourhoodTree(unittest.TestCase):
    """
    Tests for the PixelNeighbourhoodTree class.
    """

    def test_query(self):
        """
        Tests the PixelNeighbourhoodTree.query() function.
        """
        image = Image()
        image.create(1, 1)
        tree = PixelNeighbourhoodTree(image, 2, np.full((1, 1), True))

        tests = [
            (
                np.arange(12).reshape(2, 2, 3),
                4,
                np.full((1, 1), True),
                np.arange(12).reshape(4, 3),
                [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
            ), (
                np.arange(12).reshape(2, 2, 3),
                2,
                np.full((1, 1), True),
                np.arange(12).reshape(4, 3),
                [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
            ), (
                np.arange(48).reshape(4, 4, 3),
                2,
                np.full((1, 1), True),
                np.arange(48).reshape(16, 3),
                [Point(x, y) for y in range(4) for x in range(4)]
            ), (
                np.arange(12).reshape(2, 2, 3),
                2,
                np.full((1, 1), True),
                [[0, 0, 0], [3, 5, 4], [6, 7, 7], [99, 99, 99]],
                [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
            )
        ]
        for i, (pixels, branching_factor, neighbours, queries, want_points) in enumerate(tests):
            src_image = Image()
            src_image.create(pixels.shape[0], pixels.shape[1])
            src_image[:, :] = pixels
            tree = PixelNeighbourhoodTree(src_image, branching_factor, neighbours)
            for j, query in enumerate(queries):
                out_image = Image()
                out_image.create(1, 1)
                out_image[:, :] = query
                have = tree.query(out_image, Point(0, 0))
                want = want_points[j]
                self.assertEqual(have, want, f'Test {i}.{j}: Have = {have}, Want = {want}.')

    def test_erase_empty_neighbourhoods(self):
        """
        Tests the PixelNeighbourhoodTree.__erase_empty_neighbourhoods() function.
        """
        image = Image()
        image.create(1, 1)
        tree = PixelNeighbourhoodTree(image, 2, np.full((1, 1), True))

        tests = [
            # Full Neighbourhoods
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            ), (
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0])
            ), (
                np.array([0]),
                np.array([0, 1]),
                np.array([0]),
                np.array([0, 1])
            ), (
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1])
            ), (
                np.array([0, 1, 2]),
                np.array([2, 0, 1, 1]),
                np.array([0, 1, 2]),
                np.array([2, 0, 1, 1])
            ),
            # Empty Neighbourhoods
            (
                np.array([0]),
                np.array([1]),
                np.array([]),
                np.array([0])
            ), (
                np.array([0, 1]),
                np.array([1]),
                np.array([1]),
                np.array([0])
            ), (
                np.array([0, 1, 2]),
                np.array([0, 2]),
                np.array([0, 2]),
                np.array([0, 1])
            ), (
                np.array([0, 1, 2]),
                np.array([1, 1]),
                np.array([1]),
                np.array([0, 0])
            ),
        ]
        for i, (neighbourhoods, partition, want_neighbourhoods, want_partition) in enumerate(tests):
            have_neighbourhoods = tree._PixelNeighbourhoodTree__erase_empty_neighbourhoods(neighbourhoods, partition)
            self.assertTrue(np.all(have_neighbourhoods == want_neighbourhoods), f'Test {i} [Neighbourhoods]: Have = {have_neighbourhoods}, Want = {want_neighbourhoods}.')
            self.assertTrue(np.all(partition == want_partition), f'Test {i} [Partition]: Have = {partition}, Want = {want_partition}.')

    def test_extract_neighbourhood(self):
        """
        Tests the PixelNeighbourhoodTree.__extract_neighbourhood() function.
        """
        image = Image()
        image.create(4, 4)
        tree = PixelNeighbourhoodTree(image, 100, np.full((3, 3), True))

        tests = [
            (np.zeros(image.shape), Point(2, 0), np.zeros(27)),
            (np.full(image.shape, [0, 1, 2]), Point(0, 0), np.concatenate([[i] * 9 for i in range(3)], axis=0)),
            (np.arange(3 * image.area).reshape(image.shape), Point(1, 1), [0, 3, 6, 12, 15, 18, 24, 27, 30] + [1, 4, 7, 13, 16, 19, 25, 28, 31] + [2, 5, 8, 14, 17, 20, 26, 29, 32])
        ]
        for i, (pixels, point, want) in enumerate(tests):
            image[:, :] = pixels
            have = tree._PixelNeighbourhoodTree__extract_neighbourhood(image, point)
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')

    def test_average_neighbourhood(self):
        """
        Tests the PixelNeighbourhoodTree.__average_neighbourhood() function.
        """
        image = Image()
        image.create(4, 4)
        image[:, :] = np.arange(3 * image.area).reshape(image.shape)
        tree = PixelNeighbourhoodTree(image, 100, np.full((1, 1), True))

        tests = [
            ([Point(0, 0)], [0, 1, 2]),
            ([Point(0, 0), Point(0, 0)], [0, 1, 2]),
            ([Point(0, 0), Point(1, 0), Point(2, 0)], [3, 4, 5]),
            ([Point(0, 1), Point(0, 0)], [6, 7, 8]),
            (list(image.cover()), [22.5, 23.5, 24.5]),
        ]
        for i, (points, want) in enumerate(tests):
            have = tree._PixelNeighbourhoodTree__average_neighbourhood(points)
            self.assertTrue(np.allclose(have, want), f'Test {i}: Have = {have}, Want = {want}.')

    def test_partition_points(self):
        """
        Tests the PixelNeighbourhoodTree.__partition_points() function.
        """
        image = Image()
        image.create(4, 4)
        image[:, :] = np.arange(3 * image.area).reshape(image.shape)
        tree = PixelNeighbourhoodTree(image, 100, np.full((1, 1), True))

        tests = [
            (
                [],
                [],
                []
            ), (
                [[0, 0, 0]],
                [Point(0, 0)],
                [0]
            ), (
                [[0, 1, 2], [3, 4, 5]],
                [Point(0, 0)],
                [0]
            ), (
                [[0, 1, 2], [3, 4, 5], [3, 4, 5]],
                [Point(1, 0)],
                [1]
            ), (
                [[0, 1, 2]],
                [Point(0, 0), Point(1, 1)],
                [0]
            ), (
                [[0, 1, 2], [3, 4, 5]],
                [Point(0, 0), Point(1, 0)],
                [0, 1]
            ), (
                [[0, 1, 2], [3, 4, 5]],
                [Point(1, 0), Point(0, 0)],
                [1, 0]
            ), (
                [[0, 1, 2], [12, 13, 14], [24, 25, 26], [36, 37, 38]],
                list(image.cover('raster')),
                3 * [0] + 4 * [1] + 4 * [2] + 5 * [3]
            )
        ]
        for i, (neighbourhoods, points, want) in enumerate(tests):
            have = tree._PixelNeighbourhoodTree__partition_points(neighbourhoods, points)
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')
