import unittest
import math
from vector import Vector

class TestVector(unittest.TestCase):
    def setUp(self):
        """Set up test vectors before each test."""
        self.v1 = Vector([1.0, 2.0, 3.0])
        self.v2 = Vector([4.0, 5.0, 6.0])
        self.v3 = Vector([0.0, 0.0, 0.0])
        self.v4 = Vector([1.0, 0.0])

    def test_init(self):
        """Test Vector initialization."""
        self.assertEqual(self.v1.values, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            Vector([])  # Empty vector

    def test_repr(self):
        """Test Vector string representation."""
        self.assertEqual(repr(self.v1), "Vector([1.0, 2.0, 3.0])")

    def test_len(self):
        """Test Vector length."""
        self.assertEqual(len(self.v1), 3)
        self.assertEqual(len(self.v4), 2)

    def test_getitem(self):
        """Test Vector indexing."""
        self.assertEqual(self.v1[0], 1.0)
        self.assertEqual(self.v1[2], 3.0)
        with self.assertRaises(IndexError):
            _ = self.v1[3]

    def test_add(self):
        """Test Vector addition."""
        result = self.v1 + self.v2
        self.assertEqual(result.values, [5.0, 7.0, 9.0])
        with self.assertRaises(ValueError):
            _ = self.v1 + self.v4  # Different lengths

    def test_sub(self):
        """Test Vector subtraction."""
        result = self.v2 - self.v1
        self.assertEqual(result.values, [3.0, 3.0, 3.0])
        with self.assertRaises(ValueError):
            _ = self.v1 - self.v4  # Different lengths

    def test_mul(self):
        """Test scalar multiplication."""
        result = self.v1 * 2.0
        self.assertEqual(result.values, [2.0, 4.0, 6.0])
        result = 2.0 * self.v1  # Test __rmul__
        self.assertEqual(result.values, [2.0, 4.0, 6.0])

    def test_eq(self):
        """Test Vector equality."""
        self.assertEqual(Vector([1.0, 2.0, 3.0]), self.v1)
        self.assertNotEqual(self.v1, self.v2)

    def test_neg(self):
        """Test Vector negation."""
        result = -self.v1
        self.assertEqual(result.values, [-1.0, -2.0, -3.0])

    def test_dot(self):
        """Test dot product."""
        result = self.v1.dot(self.v2)
        self.assertEqual(result, 32.0)  # 1*4 + 2*5 + 3*6 = 32
        with self.assertRaises(ValueError):
            self.v1.dot(self.v4)  # Different lengths

    def test_norm(self):
        """Test Vector norm."""
        result = self.v1.norm()
        self.assertAlmostEqual(result, math.sqrt(14.0))  # sqrt(1^2 + 2^2 + 3^2)
        self.assertEqual(self.v3.norm(), 0.0)

    def test_normalize(self):
        """Test Vector normalization."""
        result = self.v1.normalize()
        norm = math.sqrt(14.0)
        expected = [1.0/norm, 2.0/norm, 3.0/norm]
        for a, b in zip(result.values, expected):
            self.assertAlmostEqual(a, b)
        with self.assertRaises(ValueError):
            self.v3.normalize()  # Zero vector

    def test_project_onto(self):
        """Test Vector projection."""
        v1 = Vector([3.0, 4.0])
        v2 = Vector([1.0, 0.0])
        result = v1.project_onto(v2)
        self.assertEqual(result.values, [3.0, 0.0])  # Projection should be [3, 0]
        with self.assertRaises(ValueError):
            self.v1.project_onto(self.v3)  # Zero vector
        with self.assertRaises(ValueError):
            self.v1.project_onto(self.v4)  # Different lengths

    def test_angle_with(self):
        """Test angle between vectors."""
        v1 = Vector([1.0, 0.0])
        v2 = Vector([0.0, 1.0])
        result = v1.angle_with(v2)
        self.assertAlmostEqual(result, math.pi/2)  # 90 degrees
        with self.assertRaises(ValueError):
            self.v1.angle_with(self.v3)  # Zero vector
        with self.assertRaises(ValueError):
            self.v1.angle_with(self.v4)  # Different lengths

if __name__ == '__main__':
    unittest.main()