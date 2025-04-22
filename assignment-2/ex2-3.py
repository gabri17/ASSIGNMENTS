import numpy as np
import random
import matplotlib.pyplot as plt

A = 8.8480182
def f(x):
    return (1/A) * (x**2) * (np.sin(np.pi*x)**2)

max_point = 0.8

samples = []
N = 20_000
n = 200
confidence = 0.95

for i in range(N):
    U1 = random.uniform(-3, 3)
    U2 = random.uniform(0, max_point)
    
    while U2 >= f(U1):
        U1 = random.uniform(-3, 3)
        U2 = random.uniform(0, max_point)
    
    samples.append(U1)

print("\t\t", "Regular procedure for large datasets\n")
#CI for median
etha = 1.96
p = 0.5
j = np.floor(n*p-etha*np.sqrt(n*p*(1-p)))
k = np.ceil(n*p+etha*np.sqrt(n*p*(1-p))) + 1
"""sorted_samples = samples
sorted_samples.sort()
"""
sorted_n = samples[0:n]
sorted_n.sort()
#print(j, k)
print("Median: ")
print("\t", sorted_n[int(j)], sorted_n[int(k)])
#print(sorted_n[0], sorted_n[n-1])
#print(sorted_samples[0], sorted_samples[N-1])

#CI for 0.9 quantile
etha = 1.96
p = 0.9
j = np.floor(n*p-etha*np.sqrt(n*p*(1-p)))
k = np.ceil(n*p+etha*np.sqrt(n*p*(1-p))) + 1
sorted_n = samples[0:n]
sorted_n.sort()
#print(j, k)
print("0.9 quantile: ")
print("\t", sorted_n[int(j)], sorted_n[int(k)])

#CI for mean - using asymptotic case: we have no heavy tail (?)
etha = 1.96
first_n = samples[0:n]
empirical_mean = sum(first_n) / n
empirical_std_dev = sum([(x - empirical_mean)**2 for x in first_n]) / n
print("Mean: ")
delta = etha * (np.sqrt(empirical_std_dev / n))
print("\t", empirical_mean - delta, empirical_mean + delta)

print("\n\t\tBootstrap procedure\n")

#Bootstrap procedure...
R = 999 #standard for confidence = 0.95

#median
first_n = samples[0:n] #use dataset of 200 elements
bootstrap_sample = []
medians = []
means = []
percentiles_90 = []

for i in range(R):

    bootstrap_sample = []
    
    for j in range(n):
        bootstrap_sample.append(random.choice(first_n))
    
    bootstrap_sample.sort()
    medians.append(0.5*(bootstrap_sample[int(np.floor(n/2))]+bootstrap_sample[int(np.floor(n/2))+1]))
    k1 = np.floor(n*0.9+0.1)
    k2 = np.ceil(n*0.95+0.1)
    percentiles_90.append(0.5*(bootstrap_sample[int(k1)]+bootstrap_sample[int(k2)]))
    means.append(sum(bootstrap_sample) / n)

medians.sort()
means.sort()
percentiles_90.sort()
print("Median: ")
print("\t", medians[25], medians[975])
print("0.9 quantile: ")
print("\t", percentiles_90[25], percentiles_90[975])
print("Mean: ")
print("\t", means[25], means[975])

plt.figure()
#plt.boxplot(samples[0:n])
plt.violinplot(samples[0:n], showmeans=True, showmedians=True)
plt.show()