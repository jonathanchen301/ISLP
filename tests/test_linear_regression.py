import numpy as np
from linear_regression.least_squares import *

class TestResidual:
    
    def test_case_1(self):
        actual = np.array([5, 10, 15])
        pred = np.array([3, 8, 10])
        expected = np.array([2, 2, 5])
        result = residual(actual, pred)
        np.testing.assert_array_equal(result, expected)

    def test_case_2(self):
        actual = np.array([1, 2, 3])
        pred = np.array([1, 2, 3])
        expected = np.array([0, 0, 0])
        result = residual(actual, pred)
        np.testing.assert_array_equal(result, expected)

    def test_case_3(self):
        actual = np.array([3, 6, 9])
        pred = np.array([4, 7, 10])
        expected = np.array([-1, -1, -1])
        result = residual(actual, pred)
        np.testing.assert_array_equal(result, expected)

class TestRSS:

    def test_case_1(self):
        actual = np.array([5, 10, 15])
        pred = np.array([3, 8, 10])
        expected = np.array([4, 4, 25])
        result = residual_sum_of_squares(actual, pred)
        np.testing.assert_array_equal(result, np.sum(expected))

    def test_case_2(self):
        actual = np.array([1, 2, 3])
        pred = np.array([1, 2, 3])
        expected = np.array([0, 0, 0])
        result = residual_sum_of_squares(actual, pred)
        np.testing.assert_array_equal(result, np.sum(expected))

    def test_case_3(self):
        actual = np.array([3, 6, 9])
        pred = np.array([4, 7, 10])
        expected = np.array([1, 1, 1])
        result = residual_sum_of_squares(actual, pred)
        np.testing.assert_array_equal(result, np.sum(expected))

class TestLeastSquaresCoefficient:

    def test_case_1(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([2, 4, 6, 8, 10])
        expected_beta0 = 0
        expected_beta1 = 2
        beta0, beta1 = least_squares_coefficient(X, Y)
        assert np.isclose(beta0, expected_beta0), f"Expected {expected_beta0}, but got {beta0}"
        assert np.isclose(beta1, expected_beta1), f"Expected {expected_beta1}, but got {beta1}"

    def test_case_2(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([2.1, 3.9, 6.2, 8.1, 9.9])
        expected_beta0 = 0.1
        expected_beta1 = 2
        beta0, beta1 = least_squares_coefficient(X, Y)
        assert np.isclose(beta0, expected_beta0, atol=0.1), f"Expected {expected_beta0}, but got {beta0}"
        assert np.isclose(beta1, expected_beta1, atol=0.1), f"Expected {expected_beta1}, but got {beta1}"

    def test_case_3(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([3, 3, 3, 3, 3])
        expected_beta0 = 3
        expected_beta1 = 0
        beta0, beta1 = least_squares_coefficient(X, Y)
        assert np.isclose(beta0, expected_beta0), f"Expected {expected_beta0}, but got {beta0}"
        assert np.isclose(beta1, expected_beta1), f"Expected {expected_beta1}, but got {beta1}"