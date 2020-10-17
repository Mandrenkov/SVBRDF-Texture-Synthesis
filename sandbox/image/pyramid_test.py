import numpy as np  # type: ignore
import unittest

from . import Image, Pyramid


class TestPyramid(unittest.TestCase):
    """
    Tests for the Pyramid class.
    """

    def test_init(self):
        """
        Tests the Pyramid.__init__() function.
        """
        tests = [
            ((8, 8), 1, [(8, 8)]),
            ((8, 8), 2, [(8, 8), (4, 4)]),
            ((8, 8), 3, [(8, 8), (4, 4), (2, 2)]),
            ((8, 8), 4, [(8, 8), (4, 4), (2, 2), (1, 1)]),
        ]
        for i, ((width, height), levels, want_layers) in enumerate(tests):
            image = Image()
            image.create(width, height)
            pyramid = Pyramid(image, levels)

            have_levels = pyramid._Pyramid__levels
            have_layers = list(map(lambda img: img.size, pyramid._Pyramid__layers))
            self.assertEqual(have_levels, levels, f'Test {i} [Levels]: Have = {have_levels}, Want = {levels}.')
            self.assertEqual(have_layers, want_layers, f'Test {i} [Layers]: Have = {have_layers}, Want = {want_layers}.')

    def test_reconstruct(self):
        """
        Tests the Pyramid.reconstruct() function.
        """
        base = Image()
        base.create(8, 8)
        base[:, :] = np.arange(8 * 8 * 3).reshape(8, 8, 3)

        tests = [1, 2, 3]
        for i, levels in enumerate(tests):
            pyramid = Pyramid(base, levels)
            recon = pyramid.reconstruct()
            self.assertTrue(base == recon, f'Test {i}: Have = {recon[:, :]}, Want = {base[:, :]}.')

    def test_items(self):
        """
        Tests the Pyramid.__getitem__() and Pyramid.__setitem__() functions.
        """
        def make_image(width: int, height: int, brightness: int) -> Image:
            """Returns a greyscale Image with the given width, height, and brightness."""
            image = Image()
            image.create(width, height)
            image[:, :] = np.full((height, width, 3), brightness)
            return image

        base = Image()
        base.create(16, 16)
        pyramid = Pyramid(base, 3)

        tests = [
            (0, base, 0, base),
            (0, base, 1, make_image(8, 8, 0)),
            (0, base, 2, make_image(4, 4, 0)),
            (2, make_image(4, 4, 1), 2, make_image(4, 4, 1)),
            (1, make_image(6, 6, 2), 1, make_image(6, 6, 2)),
        ]
        for i, (set_key, image, get_key, want) in enumerate(tests):
            pyramid[set_key] = image
            have = pyramid[get_key]
            self.assertTrue(have == want, f'Test {i}: Have = {have[:, :]}, Want = {want[:, :]}.')

