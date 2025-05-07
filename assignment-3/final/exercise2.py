import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

##### DATA PREPARATION
data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

##### LEAST SQUARES POLYNOMIAL FITTING with plotting

errors = [] #mean squared errors
M = 10 #max degree of polynomial

plt.figure(figsize=(16, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
plt.subplots_adjust(wspace=0.3)

#colormap for better line distinction
colors = [
    "#e6194b",  
    "#3cb44b",  
    "#ffe119",  
    "#0082c8",  
    "#f58231",  
    "#911eb4",  
    "#46f0f0",  
    "#f032e6",  
    "#d2f53c",  
    "#fabebe"   
]

for i in range(1, M+1):
    coeffs = np.polyfit(x, y, i)
    
    #print the coefficients of the polynomial
    print(f"Degree {i} Coefficients:", coeffs)

    #calculate the polynomial values
    yy = np.zeros(len(x))
    for j in range(len(coeffs)):
        k = len(coeffs) - j - 1
        yy += coeffs[j] * (x**k) #first coefficient has the highest degree

    errors.append(np.mean((y - yy)**2))

    ax1.plot(x, yy, label=f'Degree {i}', linewidth=2, color=colors[i - 1])

ax1.scatter(x, y, color='gray', alpha=0.5)

ax1.set_title("Time vs Metric", fontsize=14)
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Metric", fontsize=12)

ax1.set_xticks(ax1.get_xticks())
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(min(y)-1, max(y)+1)
ax1.legend(title="Polynomial Degrees")

#We can graphically see that any polymonial of degree higher than 5 is enough to fit the data

for i in range(len(errors)):
    print(f"Mean Squared Error for degree {i+1}: {errors[i]}")

### FIND THE OPTIMAL DEGREE
ax2.plot(range(1, M+1), errors, marker='o', linestyle='-', color='darkred', linewidth=2)
ax2.set_title("MSE vs Polynomial Degree", fontsize=14)
ax2.set_xlabel("Polynomial Degree", fontsize=12)
ax2.set_ylabel("Mean Squared Error", fontsize=12)
ax2.set_xticks(range(1, M+1))

#We can see it is 5 since after that the error is not decreasing significantly
plt.tight_layout()
plt.show()
