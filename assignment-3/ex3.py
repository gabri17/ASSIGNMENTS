import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

m = 5 #best degree

p = np.polyfit(x, y, m)
yy = np.zeros(len(x))
for j in range(len(p)):
    k = len(p) - j - 1
    print(k, p[j])
    yy += p[j] * (x**k)

residuals = y - yy

#so let's compute the residuals by subtracting the polynomial of degree 5 from the data
fig = plt.figure()
grid = plt.GridSpec(2, 2)
ax_old1 = plt.subplot(grid[0, 0])
ax_old2 = plt.subplot(grid[1, 0]) 
ax_new3 = plt.subplot(grid[0, 1])
ax_new4 = plt.subplot(grid[1, 1])


ax_old1.scatter(x, y, color='gray', s=20, alpha=0.7)
ax_old1.set_title('Scatter plot of old distribution')
ax_old1.set_ylim(min(y)-1, max(y)+1)

BINS=200

ax_old2.hist(y, bins=BINS, color='blue', alpha=0.7, density=True)
ax_old2.set_title('Histogram of old distribution')


ax_new3.scatter(x, residuals, color='black', s=20, alpha=0.7)
ax_new3.set_title('Scatter plot of new distribution')
ax_new3.set_ylim(min(residuals)-1, max(residuals)+1)

ax_new4.hist(residuals, bins=BINS, color='red', alpha=0.7, density=True)


def f(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )


points = np.linspace(-11, 11, 1000)
mu1, sigma1 = -5, 3
mu2, sigma2 = 0, 6
mu3, sigma3 = 4, 1

ax_new4.plot(points, list(map(lambda x: f(x, mu1, sigma1), points)), color='green')
ax_new4.plot(points, list(map(lambda x: f(x, mu2, sigma2), points)), color='orange')
ax_new4.plot(points, list(map(lambda x: f(x, mu3, sigma3), points)), color='purple')

""" ax_old2.plot(points, list(map(lambda x: f(x, mu1, sigma1), points)), color='green')
ax_old2.plot(points, list(map(lambda x: f(x, mu2, sigma2), points)), color='orange')
ax_old2.plot(points, list(map(lambda x: f(x, mu3, sigma3), points)), color='purple')
ax_old2.set_ylim(0, 0.06) """


ax_new4.set_title('Histogram of new distribution')

plt.tight_layout()
plt.show()

#EX4: must be done on data with or without trend?