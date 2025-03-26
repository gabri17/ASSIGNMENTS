import numpy as np

mean_1, var_1 = -2, 2
mean_2, var_2 = 4, 1
mean_3, var_3 = 10, 3
mean_4, var_4 = 15, 2

p_1 = 0.15
p_2 = 0.25
p_3 = 0.35
p_4 = 0.2

N = 1_000_000

s_1 = np.random.normal(mean_1, np.sqrt(var_1), N)
s_2 = np.random.normal(mean_2, np.sqrt(var_2), N)
s_3 = np.random.normal(mean_3, np.sqrt(var_3), N)
s_4 = np.random.normal(mean_4, np.sqrt(var_4), N)

mean_dataset_1, var_dataset_1 = np.mean(s_1), np.var(s_1)

#print(s.shape)

#Var(X) = E[Var(X|Y)] + Var(E[X|Y])