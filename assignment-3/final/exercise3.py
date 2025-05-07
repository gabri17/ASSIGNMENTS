import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##### DATA PREPARATION
data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

##### REMOVE TREND

m = 5 #best degree to fit the trend

p = np.polyfit(x, y, m)
yy = np.zeros(len(x))
for j in range(len(p)):
    k = len(p) - j - 1
    yy += p[j] * (x**k)

#so let's compute the residuals by subtracting the polynomial of degree 5 from the data
residuals = y - yy

##### PLOT RESULTS

BINS=200

fig = plt.figure()
grid = plt.GridSpec(2, 2)
ax_old1 = plt.subplot(grid[0, 0])
ax_old2 = plt.subplot(grid[1, 0]) 
ax_new3 = plt.subplot(grid[0, 1])
ax_new4 = plt.subplot(grid[1, 1])


ax_old1.scatter(x, y, color='gray', s=20, alpha=0.3)
ax_old1.set_title('Scatter plot of data WITH trend')
ax_old1.set_ylim(min(y)-1, max(y)+1)
ax_old1.set_xlabel("Time")
ax_old1.set_ylabel("Metric")

ax_old2.hist(y, bins=BINS, color='blue', alpha=0.5, density=False)
ax_old2.set_xlabel("Metric values")
ax_old2.set_ylabel("Frequency")
ax_old2.set_title('Histogram of data WITH trend')


ax_new3.scatter(x, residuals, color='violet', s=20, alpha=0.3)
ax_new3.set_title('Scatter plot of data WITHOUT trend')
ax_new3.set_ylim(min(residuals)-1, max(residuals)+1)
ax_new3.set_xlabel("Time")
ax_new3.set_ylabel("Metric")

ax_new4.hist(residuals, bins=BINS, color='red', alpha=0.5, density=False)
ax_new4.set_title('Histogram of of data WITHOUT trend')
ax_new4.set_xlabel("Metric values")
ax_new4.set_ylabel("Frequency")

plt.tight_layout()
plt.show()