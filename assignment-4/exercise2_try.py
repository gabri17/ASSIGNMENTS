import numpy as np
import matplotlib.pyplot as plt
from exercise1_try import MM1QueueSimulator, confidence_interval

arrival_rate = 1
service_rate = 2
simulation_time = 500

replications = 10

emp_X = []
emp_Y = []

averages_stratium = []
queue_lengths = {}

for j in range(replications):
    simulator = MM1QueueSimulator(arrival_rate=arrival_rate, service_rate=service_rate, simulation_time=simulation_time)
    simulator.simulate()


    X = simulator.compute_average_time_in_system()
    Y = simulator.compute_average_in_queue()
    emp_X.append(X)
    emp_Y.append(Y)

    vett = simulator.averages_time_in_sys_and_queue_length()

    for (q_length, time) in vett:
        if(queue_lengths.get(q_length) == None):
            queue_lengths[q_length] = (time, 1)
        else:
            queue_lengths[q_length][0] = (queue_lengths[q_length][0]*queue_lengths[q_length][1] + time) / (queue_lengths[q_length][1] + 1)
            queue_lengths[q_length][1] += 1

n = 0
for _, v in queue_lengths.items():
    ni, _ = v
    n += ni

partial_avg = 0
for k, v in queue_lengths.items():
    ni, avg = v
    partial_avg += (ni / n) * avg

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
print(f"Post-stratium: {(partial_avg)}")
