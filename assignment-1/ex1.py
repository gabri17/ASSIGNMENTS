import numpy as np
import random

mean_1, var_1 = -2, 2
mean_2, var_2 = 4, 1
mean_3, var_3 = 10, 3
mean_4, var_4 = 15, 2

p_1 = 0.15
f_1 = 0.15
p_2 = 0.25
f_2 = 0.40
p_3 = 0.35
f_3 = 0.75
p_4 = 0.25
f_4 = 1

N = 1_000_000

samples = []

for i in range(N):
    U = random.random()

    if(U <= f_1):
        n = np.random.normal(mean_1, np.sqrt(var_1), 1)
    elif (U <= f_2):
        n = np.random.normal(mean_2, np.sqrt(var_2), 1)
    elif (U <= f_3):
        n = np.random.normal(mean_3, np.sqrt(var_3), 1)
    else:
        n = np.random.normal(mean_4, np.sqrt(var_4), 1)

    samples.append(float(n[0]))

#print(samples)
empirical_mean_dataset, empirical_var_dataset = np.mean(samples), np.var(samples)
print("Dataset mean: " + str(empirical_mean_dataset), "\nDataset variance: " + str(empirical_var_dataset))

computed_mean = mean_1 * p_1 + mean_2 * p_2 + mean_3 * p_3 + mean_4 * p_4
print("Computed mean: " + str(computed_mean))

#Var(X) = E[Var(X|Y)] + Var(E[X|Y])
#X is my conditional distribution
#Y is the probability of each gaussian
#X | Y is one of the gaussians