import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

plt.scatter(x, y, color='gray', s=20, alpha=0.7)
errors = []

M = 5 #max degree of polynomial
for i in range(1, M+1):
    m = np.polyfit(x, y, i)
    print(i, m)
    yy = np.zeros(len(x))
    for j in range(len(m)):
        k = len(m) - j - 1
        print(k, m[j])
        yy += m[j] * (x**k)

    errors.append(np.mean((y - yy)**2))

    plt.plot(x, yy, label=f'Degree {i}')

plt.ylim(min(y)-1, max(y)+1)
plt.legend()
plt.tight_layout()
plt.show()

for i in range(len(errors)):
    print(f"Mean Squared Error for degree {i+1}: {errors[i]}")

m = errors.index(min(errors)) + 1
print(f"Best degree: {m}") #but already at degree 4/5 we have enough good results
