import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data_file = 'assignment-3/data_ex1_wt.csv'
try:
    df = pd.read_csv(data_file, header=None, names=['time', 'metric'])
except FileNotFoundError:
    data_file = 'data_ex1_wt.csv'
    df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

residuals = df['metric'].values.reshape(-1, 1)  

k_values = range(1, 6)
n_init = 10
aic, bic, log_likelihoods = [], [], []

for k in k_values:
    gmm = GaussianMixture(n_components=k, n_init=n_init, random_state=42)
    gmm.fit(residuals)
    
    log_likelihoods.append(gmm.score(residuals))  
    aic.append(gmm.aic(residuals))                
    bic.append(gmm.bic(residuals))              

optimal_k_aic = k_values[np.argmin(aic)]
optimal_k_bic = k_values[np.argmin(bic)]

plt.figure(figsize=(10, 5))
plt.plot(k_values, aic, marker='o', label='AIC')
plt.plot(k_values, bic, marker='o', label='BIC')
plt.xlabel('Number of Gaussians (k)')
plt.ylabel('Criterion Value')
plt.title('AIC/BIC vs. Number of Components')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal k (AIC): {optimal_k_aic}")
print(f"Optimal k (BIC): {optimal_k_bic}")

gmm_k3 = GaussianMixture(n_components=3, n_init=n_init, random_state=42)
gmm_k3.fit(residuals)

print("\nEstimated Parameters (k=3):")
print(f"Means: {gmm_k3.means_.flatten()}")
print(f"Variances: {gmm_k3.covariances_.flatten()}")

TRUE_MEANS = [-5, 0, 4]
TRUE_VARIANCES = [3, 6, 1]
print("\nGround Truth:")
print(f"Means: {TRUE_MEANS}")
print(f"Variances: {TRUE_VARIANCES}")