import unittest

from . import Point


class TestPoint(unittest.TestCase):
    """
    Tests for the Point class.
    """

    def test_properties(self):
        """
        Tests the Point constructor and properties.
        """
        tests = [
            (Point(0, 0), 0, 0),
            (Point(1, 2), 1, 2),
        ]
        for i, (point, want_x, want_y) in enumerate(tests):
            have_x, have_y = point.x, point.y
            self.assertEqual(have_x, want_x, f'Test {i}: Have = {have_x}, Want = {want_x}.')
            self.assertEqual(have_y, want_y, f'Test {i}: Have = {have_y}, Want = {want_y}.')

    def test_hash(self):
        """
        Tests the Point.__hash__() function.
        """
        tests = [
            (Point(0, 0), hash((0, 0))),
            (Point(1, 2), hash((1, 2))),
        ]
        for i, (point, want) in enumerate(tests):
            have = hash(point)
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_add(self):
        """
        Tests the Point.__add__() function.
        """
        tests = [
            (Point(0, 0), Point(1, 2), Point(1, 2)),
            (Point(3, 4), Point(0, 0), Point(3, 4)),
            (Point(5, 6), Point(4, 3), Point(9, 9)),
        ]
        for i, (point_1, point_2, want) in enumerate(tests):
            have = point_1 + point_2
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_sub(self):
        """
        Tests the Point.__sub__() function.
        """
        tests = [
            (Point(1, 2), Point(0, 0), Point(1, 2)),
            (Point(3, 3), Point(1, 2), Point(2, 1)),
        ]
        for i, (point_1, point_2, want) in enumerate(tests):
            have = point_1 - point_2
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_mul(self):
        """
        Tests the Point.__mul__() function.
        """
        tests = [
            (Point(0, 0), 0, Point(0, 0)),
            (Point(0, 0), 2, Point(0, 0)),
            (Point(1, 2), 0, Point(0, 0)),
            (Point(1, 2), 1, Point(1, 2)),
            (Point(1, 2), 3, Point(3, 6)),
        ]
        for i, (point, scalar, want) in enumerate(tests):
            have = scalar * point
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')
    
    def test_floordiv(self):
        """
        Tests the Point.__floordiv__() function.
        """
        tests = [
            (Point(0, 0), 2, Point(0, 0)),
            (Point(6, 8), 1, Point(6, 8)),
            (Point(6, 8), 2, Point(3, 4)),
        ]
        for i, (point, scalar, want) in enumerate(tests):
            have = point // scalar
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_eq(self):
        """
        Tests the Point.__eq__() function.
        """
        tests = [
            (Point(0, 0), Point(0, 0), True),
            (Point(1, 2), Point(1, 2), True),
            (Point(0, 0), Point(1, 0), False),
            (Point(0, 0), Point(1, 2), False),
            (Point(3, 4), Point(4, 3), False),
        ]
        for i, (point_1, point_2, want) in enumerate(tests):
            have = point_1 == point_2
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_lt(self):
        """
        Tests the Point.__lt__() function.
        """
        tests = [
            (Point(0, 0), Point(0, 0), False),
            (Point(1, 1), Point(1, 1), False),
            (Point(1, 1), Point(0, 2), False),
            (Point(1, 1), Point(0, 1), False),
            (Point(1, 1), Point(0, 0), False),
            (Point(1, 1), Point(1, 0), False),
            (Point(1, 1), Point(2, 0), True),
            (Point(1, 1), Point(2, 1), True),
            (Point(1, 1), Point(2, 2), True),
            (Point(1, 1), Point(1, 2), True)
        ]
        for i, (point_1, point_2, want) in enumerate(tests):
            have = point_1 < point_2
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')

    def test_repr(self):
        """
        Tests the Point.__repr__() function.
        """
        tests = [
            (Point(0, 0), '(0, 0)'),
            (Point(1, 2), '(1, 2)')
        ]
        for i, (point, want) in enumerate(tests):
            have = point.__repr__()
            self.assertEqual(have, want, f'Test {i}: Have = {have}, Want = {want}.')
