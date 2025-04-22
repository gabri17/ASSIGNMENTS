import random
import numpy as np
import matplotlib.pyplot as plt

scalar = 10 

arrival_rate = 1.5 * scalar #events per unit of time
T = 35 * scalar #time period of interest

N = int(arrival_rate * T) #number of events in time period

events_uniform = []
exponential_arrival_time = []
events_poisson = []

for i in range(N):
    U = random.uniform(0, T)
    events_uniform.append(U)
    E = np.random.exponential(arrival_rate)
    exponential_arrival_time.append(E)
    P = np.random.poisson(arrival_rate * T)
    events_poisson.append(P)

sorted_events_uni = sorted(events_uniform)
#print(sorted_events_uni)
inter_arrival_times = [sorted_events_uni[i] - sorted_events_uni[i-1] for i in range(1, len(sorted_events_uni))]
#print(inter_arrival_times)
sorted_exponential_arrival_time = sorted(exponential_arrival_time)
events_exp = []
events_exp.append(exponential_arrival_time[0])
for i in range(len(exponential_arrival_time)):
    if i == 0:
        events_exp.append(exponential_arrival_time[0])
    else:
        events_exp.append(exponential_arrival_time[i] + events_exp[i-1])

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
#ax3 = fig.add_subplot(1, 3, 3)

BINS = 20 * scalar

def exp_pdf(x, param=arrival_rate):
    return param * np.exp(-param * x)

x_points = np.arange(0, 2, 0.01)
y_points = list(map(exp_pdf, x_points))


#ax1.hist(events_uniform, bins=BINS, density=True, alpha=0.6, color='blue')
#ax1.set_title('Uniform Distribution')
ax1.hist(inter_arrival_times, bins=BINS, density=True, alpha=0.6, color='blue')
ax1.plot(x_points, y_points, color='black')
ax1.set_title('Inter arrival times from uniform distribution')

ax2.hist(events_exp, bins=BINS, density=True, alpha=0.6, color='red')
ax2.set_title('Events from inter arrival times exponentially distributed')

""" ax3.hist(events_poisson, bins=BINS, density=True, alpha=0.6, color='green')
ax3.set_title('Poisson Distribution') """


plt.show()
