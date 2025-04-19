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
sets = 100

partition = []

for i in range(N):
    U1 = random.uniform(-3, 3)
    U2 = random.uniform(0, max_point)
    
    while U2 >= f(U1):
        U1 = random.uniform(-3, 3)
        U2 = random.uniform(0, max_point)
    
    samples.append(U1)

for j in range(sets):
    random.shuffle(samples) #superfluo - useless (since they are iid)
    partition.append(random.sample(samples, n))
    samples = samples[0:len(samples)-n]

#CI for mean - using asymptotic case: we have no heavy tail (?)
confidence = 0.95
etha = 1.96

conf_inters = []

for i in range(sets):
    empirical_mean = sum(partition[i]) / n
    empirical_std_dev = sum([(x - empirical_mean)**2 for x in partition[i]]) / n
    delta = etha * (np.sqrt(empirical_std_dev / n))

    conf_inters.append((empirical_mean - delta, empirical_mean + delta))

#nicer way to plot them?
colors = ['#e63946', '#457b9d', '#1d3557']
plt.figure(figsize=(10, 4))
plt.title('Confidence intervals')
for i in range(sets):
    plt.vlines(i, conf_inters[i][0], conf_inters[i][1], color=colors[random.randint(0, 2)])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#answer: 5% so 5 CI will be wrong (since we have designed our experiment to be in a way that my CI will contain the mean 95% of the time)