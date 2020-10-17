import os
import numpy as np  # type: ignore
import unittest

from . import Image, Point


class TestImage(unittest.TestCase):
    """
    Tests for the Image class.
    """

    load_path = os.path.join(os.path.dirname(__file__), '..', 'textures', 'test.png')
    save_path = os.path.join(os.path.dirname(__file__), '..', 'textures', 'save.png')

    def test_load_save(self):
        """
        Tests the Image.load() and Image.save() functions.
        """
        image = Image(TestImage.load_path)
        image.load()

        have_pixels = image._Image__pixels
        have_filled = image._Image__filled

        want_size = np.array([3, 4])
        want_shape = np.array([3, 4, 3])
        want_pixels = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 0], [0, 0, 0], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0]],
        ])
        want_filled = np.full((3, 4), True)

        self.assertEqual(image.width, 4, f'Load [Width]: Have = {image.width}, Want = {4}.')
        self.assertEqual(image.height, 3, f'Load [Height]: Have = {image.height}, Want = {3}.')
        self.assertEqual(image.area, 12, f'Load [Area]: Have = {image.area}, Want = {12}.')
        self.assertTrue(np.all(image.size == want_size), f'Load [Size]: Have = {image.size}, Want = {want_size}.')
        self.assertTrue(np.all(image.shape == want_shape), f'Load [Shape]: Have = {image.shape}, Want = {want_shape}.')
        self.assertTrue(np.all(have_pixels == want_pixels), f'Load [Pixels]: Have = {have_pixels}, Want = {want_pixels}.')
        self.assertTrue(np.all(have_filled == want_filled), f'Load [Filled]: Have = {have_filled}, Want = {want_filled}.')

        image._Image__path = TestImage.save_path
        image.save()
        clone = Image(TestImage.save_path)
        clone.load()

        self.assertTrue(np.all(image._Image__pixels == clone._Image__pixels), f'Save [Pixels]: Have = {image._Image__pixels}, Want = {clone._Image__pixels}.')
        self.assertTrue(np.all(image._Image__filled == clone._Image__filled), f'Save [Filled]: Have = {image._Image__filled}, Want = {clone._Image__filled}.')

        os.remove(TestImage.save_path)

    def test_create(self):
        """
        Tests the Image.create() function.
        """
        tests = [
            (1, 1),
            (3, 4),
        ]
        for i, (width, height) in enumerate(tests):
            image = Image()
            image.create(width, height)

            have_pixels = image._Image__pixels
            have_filled = image._Image__filled

            want_area = width * height
            want_size = np.array([height, width])
            want_shape = np.array([height, width, 3])
            want_pixels = np.full((height, width, 3), 0)
            want_filled = np.full((height, width), False)

            self.assertEqual(image.width, width, f'Test {i} [Width]: Have = {image.width}, Want = {width}.')
            self.assertEqual(image.height, height, f'Test {i} [Height]: Have = {image.height}, Want = {height}.')
            self.assertEqual(image.area, want_area, f'Test {i} [Area]: Have = {image.area}, Want = {want_area}.')
            self.assertTrue(np.all(image.size == want_size), f'Test {i} [Size]: Have = {image.size}, Want = {want_size}.')
            self.assertTrue(np.all(image.shape == want_shape), f'Test {i} [Shape]: Have = {image.shape}, Want = {want_shape}.')
            self.assertTrue(np.all(have_pixels == want_pixels), f'Test {i} [Pixels]: Have = {have_pixels}, Want = {want_pixels}.')
            self.assertTrue(np.all(have_filled == want_filled), f'Test {i} [Filled]: Have = {have_filled}, Want = {want_filled}.')

    def test_copy(self):
        """
        Tests the Image.copy() function.
        """
        image = Image(TestImage.load_path)
        image.load()
        clone = image.copy()

        have_path = clone._Image__path
        have_pixels = clone._Image__pixels
        have_filled = clone._Image__filled

        want_path = image._Image__path
        want_pixels = image._Image__pixels
        want_filled = image._Image__filled

        self.assertEqual(have_path, want_path, f'Before [Path]: Have = {have_path}, Want = {want_path}.')
        self.assertTrue(np.all(have_pixels == want_pixels), f'Before [Pixels]: Have = {have_pixels}, Want = {want_pixels}.')
        self.assertTrue(np.all(have_filled == want_filled), f'Before [Filled]: Have = {have_filled}, Want = {want_filled}.')

        image[:, :] = 255 - image[:, :]
        self.assertFalse(np.all(have_pixels == want_pixels), f'After [Pixels]: Have = {have_pixels}, Want = {want_pixels}.')

    def test_center(self):
        """
        Tests the Image.center() function.
        """
        tests = [
            ((1, 1), Point(0, 0)),
            ((5, 5), Point(2, 2)),
            ((4, 4), Point(2, 2)),
            ((3, 8), Point(1, 4)),
        ]
        for i, ((width, height), want) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have = image.center
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_clamp(self):
        """
        Tests the Image.clamp() function.
        """
        tests = [
            ((2, 3), Point(0, 0), Point(0, 0)),
            ((2, 3), Point(1, 2), Point(1, 2)),
            ((2, 3), Point(3, 0), Point(1, 0)),
            ((2, 3), Point(0, 4), Point(0, 2)),
            ((2, 3), Point(-2, 1), Point(0, 1)),
            ((2, 3), Point(1, -2), Point(1, 0)),
        ]
        for i, ((width, height), point, want) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have = image.clamp(point)
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_corners(self):
        """
        Tests the Image.corners() function.
        """
        tests = [
            ((1, 1), Point(0, 0), Point(0, 0)),
            ((3, 4), Point(0, 0), Point(2, 3)),
        ]
        for i, ((width, height), want_min_corner, want_max_corner) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have_min_corner, have_max_corner = image.corners()
            self.assertEqual(have_min_corner, want_min_corner, f'Test {i} [Min]: Have = {have_min_corner}, Want = {want_min_corner}.')
            self.assertEqual(have_max_corner, want_max_corner, f'Test {i} [Max]: Have = {have_max_corner}, Want = {want_max_corner}.')

    def test_cover(self):
        """
        Tests the Image.cover() function.
        """
        tests = [
            ((1, 1), 'raster', [Point(0, 0)]),
            ((3, 2), 'raster', [Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(1, 1), Point(2, 1)]),
            ((1, 1), 'spiral', [Point(0, 0)]),
            ((3, 3), 'spiral', [Point(1, 1),
                                Point(0, 2), Point(1, 2), Point(2, 2), Point(2, 1), Point(2, 0), Point(1, 0), Point(0, 0), Point(0, 1)]),
            ((3, 2), 'spiral', [Point(1, 1),
                                Point(2, 1), Point(2, 0), Point(1, 0), Point(0, 0), Point(0, 1)]),
            ((1, 3), 'spiral', [Point(0, 1),
                                Point(0, 2), Point(0, 0)]),
            ((5, 5), 'spiral', [Point(2, 2),
                                Point(1, 3), Point(2, 3), Point(3, 3), Point(3, 2), Point(3, 1), Point(2, 1), Point(1, 1), Point(1, 2),
                                Point(0, 4), Point(1, 4), Point(2, 4), Point(3, 4), Point(4, 4), Point(4, 3), Point(4, 2), Point(4, 1), Point(4, 0), Point(3, 0), Point(2, 0), Point(1, 0), Point(0, 0), Point(0, 1), Point(0, 2), Point(0, 3)]),
        ]
        for i, ((width, height), mode, want) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have = [point for point in image.cover(mode)]
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_extract(self):
        """
        Tests the Image.extract() function.
        """
        image = Image(TestImage.load_path)
        image.load()
        r, g, b, k, w = list(map(np.array, [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]]))

        tests = [
            (Point(1, 1), 1, 'drop', np.array([[r, g, b], [w, k, k], [w, b, g]])),
            (Point(0, 0), 1, 'drop', np.array([[k, k, k], [k, r, g], [k, w, k]])),
            (Point(2, 2), 2, 'drop', np.array([[r, g, b, w, k], [w, k, k, w, k], [w, b, g, r, k], [k, k, k, k, k], [k, k, k, k, k]])),
            (Point(1, 1), 2, 'drop', np.array([[k, k, k, k, k], [k, r, g, b, w], [k, w, k, k, w], [k, w, b, g, r], [k, k, k, k, k]])),
            (Point(1, 1), 1, 'wrap', np.array([[r, g, b], [w, k, k], [w, b, g]])),
            (Point(0, 0), 1, 'wrap', np.array([[r, w, b], [w, r, g], [w, w, k]])),
            (Point(2, 2), 2, 'wrap', np.array([[r, g, b, w, r], [w, k, k, w, w], [w, b, g, r, w], [r, g, b, w, r], [w, k, k, w, w]])),
            (Point(1, 1), 1, 'reflect', np.array([[r, g, b], [w, k, k], [w, b, g]])),
            (Point(0, 0), 1, 'reflect', np.array([[k, w, k], [g, r, g], [k, w, k]])),
            (Point(2, 2), 2, 'reflect', np.array([[r, g, b, w, b], [w, k, k, w, k], [w, b, g, r, g], [w, k, k, w, k], [r, g, b, w, b]]))
        ]
        for i, (center, padding, mode, want) in enumerate(tests):
            window = image.extract(center, padding, mode)
            have = window._Image__pixels
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')

    def test_filled(self):
        """
        Tests the Image.filled() function.
        """
        image = Image()
        image.create(3, 3)
        image._Image__filled[:2, :2] = np.full((2, 2), True)

        tests = [
            (Point(1, 1), 1, 'drop', False, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
            (Point(0, 0), 1, 'drop', False, np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])),
            (Point(2, 2), 2, 'drop', False, np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])),
            (Point(1, 1), 1, 'wrap', False, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
            (Point(0, 0), 1, 'wrap', False, np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])),
            (Point(2, 2), 2, 'wrap', False, np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])),
            (Point(1, 1), 1, 'reflect', False, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
            (Point(0, 0), 1, 'reflect', False, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
            (Point(2, 2), 2, 'reflect', False, np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])),
            (Point(1, 1), 1, 'drop', True, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
            (Point(0, 0), 1, 'drop', True, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
            (Point(2, 2), 2, 'wrap', True, np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])),
        ]
        for i, (center, padding, mode, causal, want) in enumerate(tests):
            have = image.filled(center, padding, mode, causal)
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')

    def test_paste(self):
        """
        Tests the Image.paste() function.
        """
        r, g, b, k, w, x = list(map(np.array, [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255], [1, 1, 1]]))

        tests = [
            (Point(1, 1), 1, np.array([[x, x, x, w], [x, x, x, w], [x, x, x, r]])),
            (Point(0, 0), 1, np.array([[x, x, b, w], [x, x, k, w], [w, b, g, r]])),
            (Point(4, 1), 1, np.array([[r, g, b, x], [w, k, k, x], [w, b, g, x]])),
            (Point(2, 1), 2, np.array([[x, x, x, x], [x, x, x, x], [x, x, x, x]])),
        ]
        for i, (center, padding, want) in enumerate(tests):
            window = Image()
            window.create(2 * padding + 1, 2 * padding + 1)
            window[:, :] = 1
            image = Image(TestImage.load_path)
            image.load()
            image.paste(center, window)
            have = image._Image__pixels
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')

    def test_project(self):
        """
        Tests the Image.project() function.
        """
        tests = [
            ((2, 3), 'wrap', Point(0, 0), Point(0, 0)),
            ((2, 3), 'wrap', Point(1, 2), Point(1, 2)),
            ((2, 3), 'wrap', Point(3, 0), Point(1, 0)),
            ((2, 3), 'wrap', Point(0, 3), Point(0, 0)),
            ((2, 3), 'wrap', Point(-2, 2), Point(0, 2)),
            ((2, 3), 'wrap', Point(1, -2), Point(1, 1)),
            ((2, 3), 'reflect', Point(0, 0), Point(0, 0)),
            ((2, 3), 'reflect', Point(1, 2), Point(1, 2)),
            ((2, 3), 'reflect', Point(2, 0), Point(0, 0)),
            ((2, 3), 'reflect', Point(3, 0), Point(1, 0)),
            ((2, 3), 'reflect', Point(0, 3), Point(0, 1)),
            ((2, 3), 'reflect', Point(0, 4), Point(0, 0)),
            ((2, 3), 'reflect', Point(-1, 2), Point(1, 2)),
            ((2, 3), 'reflect', Point(1, -2), Point(1, 2)),
        ]
        for i, ((width, height), mode, point, want) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have = image.project(point, mode)
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_resize(self):
        """
        Tests the Image.resize() function.
        """
        tests = [
            ((1, 1), (1, 1)),
            ((5, 5), (3, 3)),
            ((5, 5), (7, 8)),
        ]
        for i, ((old_width, old_height), (new_width, new_height)) in enumerate(tests):
            image = Image()
            image.create(old_width, old_height)
            image.resize(new_width, new_height)
            self.assertEqual(image.width, new_width, f'Test {i} [Width]: Have = {image.width}, Want = {new_width}.')
            self.assertEqual(image.height, new_height, f'Test {i} [Height]: Have = {image.height}, Want = {new_height}.')

    def test_contains(self):
        """
        Tests the Image.__contains__() function.
        """
        tests = [
            ((2, 3), Point(0, 0), True),
            ((2, 3), Point(1, 2), True),
            ((2, 3), Point(3, 0), False),
            ((2, 3), Point(0, 4), False),
            ((2, 3), Point(-2, 1), False),
            ((2, 3), Point(1, -2), False),
        ]
        for i, ((width, height), point, want) in enumerate(tests):
            image = Image()
            image.create(width, height)
            have = point in image
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_eq(self):
        """
        Tests the Image.__eq__() function.
        """
        reference = Image()
        reference.create(2, 2)
        reference[:, :] = np.full((2, 2, 3), 1)

        tests = [
            (None, np.full((2, 2, 3), 1), np.full((2, 2), True), True),
            ("//", np.full((2, 2, 3), 1), np.full((2, 2), True), True),
            (None, np.full((2, 2, 3), 1), np.full((2, 2), False), True),
            (None, np.full((2, 2, 3), 1), np.full((3, 3), False), True),
            (None, np.full((2, 2, 3), 2), np.full((2, 2), True), False),
            (None, np.full((3, 3, 3), 1), np.full((2, 2), True), False),
        ]
        for i, (path, pixels, filled, want) in enumerate(tests):
            image = Image()
            image._Image__path = path
            image._Image__pixels = pixels
            image._Image__filled = filled
            have = reference == image
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_items(self):
        """
        Tests the Image.__getitem__() and Image.__setitem__() functions.
        """
        image = Image()
        image.create(2, 2)
        image[0, 0] = np.array([0, 0, 0])
        image[0, 1] = np.array([1, 1, 1])
        image[1, 0] = np.array([2, 2, 2])
        image[1, 1] = np.array([3, 3, 3])

        tests = [
            (Point(0, 0), np.array([0, 0, 0]), Point(0, 0), np.array([[0, 0, 0]])),
            (Point(0, 0), np.array([0, 0, 0]), (slice(1), slice(2)), np.array([[0, 0, 0], [1, 1, 1]])),
            (Point(0, 0), np.array([4, 4, 4]), Point(0, 0), np.array([[4, 4, 4]])),
            ((slice(1, 6), slice(6)), np.array([5, 5, 5]), (slice(1, 7), slice(7)), np.array([[5, 5, 5], [5, 5, 5]])),
        ]
        for i, (set_key, colour, get_key, want) in enumerate(tests):
            image[set_key] = colour
            have = image[get_key]
            self.assertTrue(np.all(have == want), f'Test {i}: Have = {have}, Want = {want}.')
