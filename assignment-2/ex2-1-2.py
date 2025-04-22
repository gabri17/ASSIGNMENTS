import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

A = 8.8480182 #se cambia??? piu alto = decade piu velocemente???
def f(x):
    if x < -3 or x > 3:
        return 0
    return (1/A) * (x**2) * (np.sin(np.pi*x)**2)

max_point = 0.8

samples = []
N = 50_000 #play with it
fails = 0

for i in range(N):
    U1 = random.uniform(-3, 3)
    U2 = random.uniform(0, max_point)
    
    while U2 >= f(U1):
        fails+=1
        U1 = random.uniform(-3, 3)
        U2 = random.uniform(0, max_point)
    
    samples.append(U1)
    #print(i, fails)

x_points = np.arange(-3, 3, 0.01)
y_points = list(map(f, x_points))

#fig, ax = plt.subplots()
#ax.set_title('Empirical PDF vs Theoretical PDF')
#ax.set_xlim([-3, 3])
#ax.hist(samples, bins=1500, density=True, alpha=0.6, color='blue')
#ax.plot(x_points, y_points, color='green')

plt.figure(figsize=(10, 4))
plt.title('Empirical PDF')
plt.hist(samples, bins=150, density=True, alpha=0.6, color='red') #play with bins
plt.plot(x_points, y_points, color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#plt.title('PDF')
#plt.plot(x_points, y_points, color='green')
#plt.show()