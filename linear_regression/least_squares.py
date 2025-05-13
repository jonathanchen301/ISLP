import numpy as np

def residual(actual, pred):
    """
    Calculate residual between actual values and observed values

    Params:
    - actual (np array): the true values
    - pred (np array): the predicted values

    Returns:
    - residual (np array): the residual
    """

    return actual - pred

def residual_sum_of_squares(actual, pred):
    """
    Calculate residual sum of squares between actual values and observed values

    Params:
    - actual (np array): the true values
    - pred (np array): the predicted values

    Returns: 
    - residual sum of squares (float): the reesidual sum of squares 
    """

    return np.sum((actual - pred)**2)

def least_squares_coefficient(X, Y):
    """
    Calculate least squares coefficient, both the slope and the intercept

    Y = beta0 + beta1X

    Params:
    - X (np array)
    - Y (np array)

    Returns:
    - beta0, beta1 (np array)
    """

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    beta1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)

    # Compute beta0 (intercept)
    beta0 = Y_mean - beta1 * X_mean

    return beta0, beta1