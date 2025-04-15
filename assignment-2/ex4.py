import numpy as np
import random

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

confidence = 0.95