import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

A = 8.8480182 
def f(x):
    if x < -3 or x > 3:
        return 0
    return (1/A) * (x**2) * (np.sin(np.pi*x)**2)

def cdf(x):
    result, _ = quad(f, -3, x)  # Integrate from -3 to x
    return result

def find_percentile(percentile):
    # Adjust CDF to solve F(x) = percentile
    def target_function(x):
        return cdf(x) - percentile

    # Use root_scalar to find the x_p
    solution = root_scalar(target_function, bracket=(-3, 3), method='bisect')
    return solution.root

percentile_value = 0.9
x_p = find_percentile(percentile_value)
print(f"The {percentile_value * 100}th percentile is approximately: {x_p:.4f}")

percentile_value = 0.5
x_p = find_percentile(percentile_value)
print(f"The {percentile_value * 100}th percentile is approximately: {x_p:.4f}")
