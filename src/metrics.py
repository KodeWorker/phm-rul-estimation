import numpy as np

def relative_percentage_error(true_failure_time, estimated_failure_time):
    return 100 * (true_failure_time - estimated_failure_time) / true_failure_time

def exponential_transformed_accuracy(error):
    if error <= 0:
        return np.exp(-np.log(0.5)*error/5)
    else:
        return np.exp(np.log(0.5)*error/20)
