import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])
print(df.info())

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="time",
    y="metric",
    s=20, 
    color="dodgerblue",
    edgecolor="black"
)

plt.title("Time vs Metric", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Metric", fontsize=12)

plt.tight_layout()
plt.xticks(rotation=45)

plt.show()
