from vector import Vector
from .operations import *

class Matrix:
    def __init__(self, values: list[list[float]]):
        if not values or not values[0]:
            raise ValueError("Matrix cannot be empty")
        self.values = values
        self.rows = len(values)
        self.cols = len(values[0])
        if any(len(row) != self.cols for row in values):
            raise ValueError("All rows must have the same number of columns")

    def __repr__(self):
        return f"Matrix({self.values})"

    def __getitem__(self, idx: int) -> list[float]:
        return self.values[idx]

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise ValueError("Matrix shapes must match")
        return Matrix([
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.values, other.values)
        ])

    def __mul__(self, other):
        if isinstance(other, Vector):
            try:
                result = mat_vec_mul(self.values, other.values)
                return Vector(result)
            except ValueError as e:
                raise ValueError("Matrix columns must match vector size") from e

        elif isinstance(other, Matrix):
            try:
                result = mat_mul(self.values, other.values)
                return Matrix(result)
            except ValueError as e:
                raise ValueError("Incompatible matrix shapes for multiplication") from e

        else:
            raise TypeError("Matrix can only be multiplied by Vector or Matrix")

    def T(self) -> 'Matrix':
        try:
            result = transpose_mat(self.values)
            return Matrix(result)
        except ValueError as e:
            raise ValueError("Matrix must not be empty") from e
        
    from operations import inverse_mat

    def inverse(self) -> 'Matrix':
        """
        Computes the inverse of the matrix using the Gauss-Jordan elimination method.
        Returns:
            Matrix: The inverse matrix.
        Raises:
            ValueError: If the matrix is not square, empty, or singular.
        """
        try:
            result = inverse_mat(self.values)
            return Matrix(result)
        except ValueError as e:
            raise ValueError(str(e)) from e

    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)
    
    def reshape(self, new_shape: tuple[int, int]) -> 'Matrix':
        """
        Reshapes the matrix to a new shape while preserving the elements in row-major order.
        Args:
            new_shape (tuple[int, int]): The desired shape (new_rows, new_cols).
        Returns:
            Matrix: The reshaped matrix.
        Raises:
            ValueError: If the matrix is empty, new_shape is invalid, or sizes are incompatible.
        """
        try:
            result = reshape_matrix(self.values, new_shape)
            return Matrix(result)
        except ValueError as e:
            raise ValueError(str(e)) from e
