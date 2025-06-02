import numpy as np
import matplotlib.pyplot as plt
from exercise1_try import MM1QueueSimulator, confidence_interval

arrival_rate = 1
service_rate = 2

replications = 10

emp_X = []
emp_Y = []

for j in range(replications):
    simulator = MM1QueueSimulator(arrival_rate=arrival_rate, service_rate=service_rate, simulation_time=500)
    simulator.simulate()

    X = simulator.compute_average_time_in_system()
    Y = simulator.compute_average_in_queue()
    emp_X.append(X)
    emp_Y.append(Y)

rho = arrival_rate / service_rate
E_Y = (rho ** 2) / (1 - rho) #average packets in queue

print(f"Queue: empirical is {np.mean(emp_Y)}, theoretical is {E_Y}")

X = np.array(emp_X)
Y = np.array(emp_Y)

cov_XY = np.cov(X, Y, ddof=1)[0, 1]
var_Y = np.var(Y, ddof=1)
c_star = - cov_XY / var_Y

X_cv = X + c_star * (Y - E_Y)
print(f"Theoretical {1/(service_rate-arrival_rate)}")
print(f"Naive: {np.mean(X)} (std dev: {np.std(X)})")
print(f"CV: {np.mean(X_cv)} (std dev: {np.std(X_cv)})")