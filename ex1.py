import random
import numpy as np
import matplotlib.pyplot as plt

scalar = 100
BINS = scalar

arrival_rate = 5 #events per unit of time
T = 50 * scalar #time period of interest

k = 1 #change to see with more/less than arrival_rate * T expected events
N = int(k*arrival_rate * T) #number of events in time period

events_uniform = []
exponential_arrival_time = []
events_poisson = []

repetitions = 1

def exp_pdf(x, param=(arrival_rate)):
    return param * np.exp(-param * x)

x_points = np.arange(0, 3, 0.01)
y_points = list(map(exp_pdf, x_points))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(x_points, y_points, color='black')
ax1.set_title('Inter arrival times from uniform distribution')
ax2.set_title('Events from inter arrival times exponentially distributed')


for j in range(repetitions):

    events_uniform = []
    exponential_arrival_time = []

    for i in range(N):
        U = random.uniform(0, T)
        events_uniform.append(U)
        E = np.random.exponential(1/arrival_rate) #parameter is arrival_rate
        exponential_arrival_time.append(E)

    f = 0
    while sum(exponential_arrival_time) > T:
        f += 1
        print(f"fail {f}")
        exponential_arrival_time = []
        for i in range(N):
            E = np.random.exponential(1/arrival_rate) #parameter is arrival_rate
            exponential_arrival_time.append(E)

    sorted_events_uni = sorted(events_uniform)
    #print(sorted_events_uni)
    inter_arrival_times = [sorted_events_uni[i] - sorted_events_uni[i-1] for i in range(1, len(sorted_events_uni))]
    #print(inter_arrival_times)
    sorted_exponential_arrival_time = sorted(exponential_arrival_time)
    events_exp = []
    for i in range(len(exponential_arrival_time)):
        if i == 0:
            events_exp.append(exponential_arrival_time[0])
        else:
            #if (exponential_arrival_time[i] + events_exp[i-1]) > T:
            #    break
            events_exp.append(exponential_arrival_time[i] + events_exp[i-1])

    #ax1.hist(events_uniform, bins=BINS, density=True, alpha=0.6, color='blue')
    #ax1.set_title('Uniform Distribution')
    ax1.hist(inter_arrival_times, bins=BINS, density=True, alpha=0.6, color='blue')
    ax2.hist(events_exp, bins=BINS // 10, density=True, alpha=0.6, color='red')

plt.show()
