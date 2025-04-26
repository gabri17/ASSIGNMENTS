import random
import numpy as np

N = 20_000
n = 200
sets = 100

A = 8.8480182 
def f(x):
    if x < -3 or x > 3:
        return 0
    return (1/A) * (x**2) * (np.sin(np.pi*x)**2)

max_point = 0.717705 #empirical computed by computing the derivative

def sampling_procedure(N):
    
    for i in range(N):
        U1 = random.uniform(-3, 3)
        U2 = random.uniform(0, max_point)
    
        while U2 >= f(U1):
            U1 = random.uniform(-3, 3)
            U2 = random.uniform(0, max_point)
    
        samples.append(U1)
    
    return samples

failures = []
REP = 30
for _ in range(REP):
    samples = sampling_procedure(N)

    global_mean = sum(samples) / len(samples)

    sets = 100
    partition = []

    for j in range(sets):
        random.shuffle(samples) #superfluo - useless (since they are iid)
        partition.append(samples[0:n]) #it chooses first n elements from samples
        samples = samples[n:] #it removes first n elements from samples

    #CI for mean - using asymptotic case: we have no heavy tail
    confidence = 0.95
    etha = 1.96

    conf_inters = []

    for i in range(sets):
        empirical_mean = sum(partition[i]) / n
        empirical_variance = sum([(x - empirical_mean)**2 for x in partition[i]]) / n
        delta = etha * (np.sqrt(empirical_variance / n))

        conf_inters.append((empirical_mean - delta, empirical_mean + delta))
    
    fails = 0
    for i in range(sets):
        a = conf_inters[i][0]
        b = conf_inters[i][1]
        if global_mean < a or global_mean > b:
            fails += 1
    
    failures.append(fails)

print("AVG of not good CIs: ", np.mean(failures), "STD: ", np.std(failures))