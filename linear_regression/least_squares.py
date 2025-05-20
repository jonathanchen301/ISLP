import numpy as np

"""
Estimating Coefficients of Simple Linear Regression.
"""

def residual(actual, pred):
    """
    Calculate residual between actual values and observed values

    Params:
    - actual (np array): the true values
    - pred (np array): the predicted values

    Returns:
    - Residual (np array): the residual
    """

    return actual - pred

def residual_sum_of_squares(actual, pred):
    """
    Calculate residual sum of squares between actual values and observed values

    Params:
    - actual (np array): the true values
    - pred (np array): the predicted values

    Returns: 
    - Residual sum of squares (float): the reesidual sum of squares 
    """

    return np.sum((actual - pred)**2)

def least_squares_coefficient(X, Y):
    """
    Calculate least squares coefficient, both the slope and the intercept

    Y = beta0 + beta1X

    Params:
    - X (np array): variable
    - Y (np array): target

    Returns:
    - beta0, beta1 (np array)
    """

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    beta1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)

    # Compute beta0 (intercept)
    beta0 = Y_mean - beta1 * X_mean

    return beta0, beta1

"""
Evaluating Accuracy of the Coefficient Estimates 
"""

def standard_error(sd, n):
    """
    Calculate the standard error of a mean.

    Params:
    - sd (float): sample standard deviation
    - sample size (int): sample size
    
    Returns:
    - Standard error of the mean (float)
    """

    return sd / np.sqrt(n)

# TODO: in later chapters, going to have to generalize to higher degrees of freedoms.
def residual_standard_error(actual, pred):
    """
    Calculate the residual standard error (RSE).
    
    Params:
    - actual (np array): the true values
    - pred (np array): the predicted values

    Returns:
    - Residual standard error (float): the residual standard error (RSE)
    """

    n = actual.shape[0]
    return np.sqrt(residual_sum_of_squares(actual, pred) / (n - 2))

def least_squares_predict(X, Y):

    """
    Calculate the predicted values using least squares.

    Params:
    - X (np array): variable
    - Y (np array): target

    Returns:
    - Predicted values (np array)
    """

    beta0, beta1 = least_squares_coefficient(X, Y)
    return beta0 + beta1 * X

def standard_error_of_slope(X, Y):
    """
    Calculate the standard error of the slope.
    
    Params:
    - X (np array): Variable
    - Y (np array): Target
    
    Returns:
    - Standard error of the slope predicted using least squares (float)
    """

    pred = least_squares_predict(X, Y)
    rse = residual_standard_error(Y, pred)
    return np.sqrt((rse ** 2) / np.sum((X - np.mean(X)) ** 2))

def standard_error_of_intercept(X, Y):
    """
    Calculate the standard error of the intercept.
    
    Params:
    - X (np array): Variable
    - Y (np array): Target
    
    Returns:
    - Standard error of the intercept predicted using least squares (float)
    """

    pred = least_squares_predict(X, Y)
    rse = residual_standard_error(Y, pred)
    return np.sqrt(((1 / X.shape[0]) + ((np.mean(X) ** 2) / np.sum((X - np.mean(X)) ** 2))) * rse ** 2)

def confidence_interval_95(X, Y):
    """
    Calculate the 95% confidence interval for the slope and intercept.
    
    Params:
    - X (np array): Variable
    - Y (np array): Target
    
    Returns:
    - Confidence interval for beta0 (tuple)
    - Confidence interval for beta1 (tuple)
    """

    beta0, beta1 = least_squares_coefficient(X, Y)
    se_beta0 = standard_error_of_intercept(X, Y)
    se_beta1 = standard_error_of_slope(X, Y)

    return (beta0 - 2 * se_beta0, beta0 + 2 * se_beta0), (beta1 - 2 * se_beta1, beta1 + 2 * se_beta1)

def t_statistic(X, Y):
    """
    Calculate the t-statistic for the slope and intercept.

    It is the number of standard deviations the predicted beta1 is away from 0.
    
    Params:
    - X (np array): Variable
    - Y (np array): Target
    
    Returns:
    - t-statistic for beta1 (float)
    """

    _, beta1 = least_squares_coefficient(X, Y)
    se_beta1 = standard_error_of_slope(X, Y)
    return beta1 / se_beta1