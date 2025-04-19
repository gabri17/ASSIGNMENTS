import random
import numpy as np
import matplotlib.pyplot as plt

arrival_rate = 3 #events per unit of time
T = 1000 #time period of interest

N = int(arrival_rate * T) #number of events in time period

events_uniform = []
events_exp = []
events_poisson = []

for i in range(N):
    U = random.uniform(0, T)
    events_uniform.append(U)
    E = np.random.exponential(1/arrival_rate)
    events_exp.append(E)
    P = np.random.poisson(arrival_rate * T)
    events_poisson.append(P)

sorted_events_uni = sorted(events_uniform)
sorted_events_exp = sorted(events_exp)

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

BINS = 120

ax1.hist(events_uniform, bins=BINS, density=True, alpha=0.6, color='blue')
ax1.set_title('Uniform Distribution')

ax2.hist(events_exp, bins=BINS, density=True, alpha=0.6, color='red')
ax2.set_title('Exponential Distribution')

ax3.hist(events_poisson, bins=BINS, density=True, alpha=0.6, color='green')
ax3.set_title('Poisson Distribution')


plt.show()
