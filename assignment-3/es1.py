import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])
print(df.info())

print(np.mean(df['metric']), np.std(df['metric']))

sns.scatterplot(df, x="time", y="metric")
plt.show()

#BINS=200
#plt.hist(df['metric'], bins=BINS, color='gray', alpha=0.7)
#plt.show()


"""
trend is:
- high times low metric
- low times high metric
"""