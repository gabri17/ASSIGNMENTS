import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

A = 8.8480182
def f(x):
    return (1/A) * (x**2) * (np.sin(np.pi*x)**2)

max_point = 0.013

samples = []
N = 100_000

for i in range(N):
    U1 = random.uniform(-3, 3)
    U2 = random.uniform(0, max_point)
    while U2 >= f(U1):
        U1 = random.uniform(-3, 3)
    samples.append(U1)

x_points = np.arange(-3, 3, 0.01)
y_points = list(map(f, x_points))


plt.figure(figsize=(10, 4))
plt.title('Empirical PDF')
plt.hist(samples, bins=1500, density=True, alpha=0.6, color='blue')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.title('PDF')
plt.plot(x_points, y_points, color='green')
plt.show()