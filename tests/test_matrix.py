import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matrix import Matrix
from vector import Vector
import unittest
import math

class TestMatrix(unittest.TestCase):
    def setUp(self):
        """Set up test matrices and vectors before each test."""
        self.m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix
        self.m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])  # 2x2 matrix
        self.m3 = Matrix([[0.0, 0.0], [0.0, 0.0]])  # 2x2 zero matrix
        self.m4 = Matrix([[1.0], [2.0]])  # 2x1 matrix
        self.m5 = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3 matrix
        self.v1 = Vector([1.0, 2.0])  # 2D vector
        self.v2 = Vector([1.0, 2.0, 3.0])  # 3D vector

    def test_init(self):
        """Test Matrix initialization."""
        self.assertEqual(self.m1.values, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(self.m1.shape(), (2, 2))
        with self.assertRaises(ValueError):
            Matrix([])  # Empty matrix
        with self.assertRaises(ValueError):
            Matrix([[1.0, 2.0], [3.0]])  # Unequal row lengths

    def test_repr(self):
        """Test Matrix string representation."""
        self.assertEqual(repr(self.m1), "Matrix([[1.0, 2.0], [3.0, 4.0]])")

    def test_getitem(self):
        """Test Matrix row indexing."""
        self.assertEqual(self.m1[0], [1.0, 2.0])
        self.assertEqual(self.m1[1], [3.0, 4.0])
        with self.assertRaises(IndexError):
            _ = self.m1[2]

    def test_add(self):
        """Test Matrix addition."""
        result = self.m1 + self.m2
        self.assertEqual(result.values, [[6.0, 8.0], [10.0, 12.0]])
        with self.assertRaises(ValueError):
            self.m1 + self.m4  # Dimension mismatch
        with self.assertRaises(TypeError):
            self.m1 + 1.0  # Invalid type

    def test_sub(self):
        """Test Matrix subtraction."""
        result = self.m2 - self.m1
        self.assertEqual(result.values, [[4.0, 4.0], [4.0, 4.0]])
        with self.assertRaises(ValueError):
            self.m1 - self.m4  # Dimension mismatch
        with self.assertRaises(TypeError):
            self.m1 - 1.0  # Invalid type

    def test_mul_matrix(self):
        """Test Matrix-Matrix multiplication."""
        # Test valid multiplication (m1 * m5: 2x2 * 2x3 = 2x3)
        result = self.m1 * self.m5
        self.assertEqual(result.values, [[9.0, 12.0, 15.0], [19.0, 26.0, 33.0]])  # [[1*1+2*4, 1*2+2*5, 1*3+2*6], [3*1+4*4, 3*2+4*5, 3*3+4*6]]
        # Test invalid multiplication (m5 * m1: 2x3 * 2x2 is invalid)
        with self.assertRaises(ValueError):
            self.m5 * self.m1  # Dimension mismatch (3 ≠ 2)
        with self.assertRaises(ValueError):
            self.m4 * self.m1  # Dimension mismatch (2x1 * 2x2)

    def test_mul_vector(self):
        """Test Matrix-Vector multiplication."""
        result = self.m1 * self.v1
        self.assertEqual(result.values, [5.0, 11.0])  # [1*1+2*2, 3*1+4*2]
        with self.assertRaises(ValueError):
            self.m1 * self.v2  # Dimension mismatch (2x2 * 3D vector)

    def test_mul_scalar(self):
        """Test Matrix-Scalar multiplication."""
        result = self.m1 * 2.0
        self.assertEqual(result.values, [[2.0, 4.0], [6.0, 8.0]])
        result = 2.0 * self.m1  # Test __rmul__
        self.assertEqual(result.values, [[2.0, 4.0], [6.0, 8.0]])
        with self.assertRaises(TypeError):
            self.m1 * "2"  # Invalid type
        with self.assertRaises(TypeError):
            "2" * self.m1  # Invalid type for __rmul__

    def test_transpose(self):
        """Test Matrix transpose."""
        result = self.m1.T()
        self.assertEqual(result.values, [[1.0, 3.0], [2.0, 4.0]])
        result = self.m4.T()
        self.assertEqual(result.values, [[1.0, 2.0]])  # 2x1 -> 1x2
        result = self.m5.T()
        self.assertEqual(result.values, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_inverse(self):
        """Test Matrix inverse."""
        result = self.m1.inverse()
        expected = [[-2.0, 1.0], [1.5, -0.5]]  # Inverse of [[1,2],[3,4]] (det=4-6=-2)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(result.values[i][j], expected[i][j])
        with self.assertRaises(ValueError):
            self.m5.inverse()  # Non-square matrix
        with self.assertRaises(ValueError):
            Matrix([[1.0, 2.0], [2.0, 4.0]]).inverse()  # Singular matrix (det=0)

    def test_shape(self):
        """Test Matrix shape."""
        self.assertEqual(self.m1.shape(), (2, 2))
        self.assertEqual(self.m4.shape(), (2, 1))
        self.assertEqual(self.m5.shape(), (2, 3))

    def test_reshape(self):
        """Test Matrix reshape."""
        result = self.m5.reshape((3, 2))
        self.assertEqual(result.values, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = self.m5.reshape((6, 1))
        self.assertEqual(result.values, [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        with self.assertRaises(ValueError):
            self.m5.reshape((2, 2))  # Incompatible size
        with self.assertRaises(ValueError):
            self.m5.reshape((0, 3))  # Invalid dimension
        with self.assertRaises(ValueError):
            self.m5.reshape((-1, 3))  # Invalid dimension

    def test_mat_mean(self):
        """Test matrix mean calculation."""
        from matrix.operations import mat_mean
        result = mat_mean(self.m1.values, 'column')
        self.assertEqual(result, [2.0, 3.0])  # [(1+3)/2, (2+4)/2]
        result = mat_mean(self.m1.values, 'row')
        self.assertEqual(result, [1.5, 3.5])  # [(1+2)/2, (3+4)/2]
        with self.assertRaises(ValueError):
            mat_mean([], 'column')  # Empty matrix

if __name__ == '__main__':
    unittest.main()