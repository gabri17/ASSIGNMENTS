import numpy as np
import matplotlib.pyplot as plt

# Parametri
lmbda = 10
mu = 15
n = 5000

# Simulazione
inter_arrival_times = np.random.exponential(1/lmbda, n)
service_times = np.random.exponential(1/mu, n)
arrival_times = np.cumsum(inter_arrival_times)

start_service_times = np.zeros(n)
departure_times = np.zeros(n)

for i in range(n):
    if i == 0:
        start_service_times[i] = arrival_times[i]
    else:
        start_service_times[i] = max(arrival_times[i], departure_times[i-1])
    departure_times[i] = start_service_times[i] + service_times[i]

delays = departure_times - arrival_times

# Funzione CDF empirica
def empirical_cdf(data):
    sorted_data = np.sort(data)
    cdf_vals = np.arange(1, len(data)+1) / len(data)
    return sorted_data, cdf_vals

# Teorica (PDF e CDF dell'esponenziale)
x = np.linspace(0, np.max(delays), 500)
pdf_theory = (mu - lmbda) * np.exp(-(mu - lmbda) * x)
cdf_theory = 1 - np.exp(-(mu - lmbda) * x)

# Plot con due subplot: PDF e CDF
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# PDF
axs[0].hist(delays, bins=50, density=True, alpha=0.6, label='Empirical PDF', color='gray')
axs[0].plot(x, pdf_theory, 'r-', lw=2, label='Theoretical PDF')
axs[0].set_title('PDF of Total Delay (M/M/1)')
axs[0].set_xlabel('Delay')
axs[0].set_ylabel('Density')
axs[0].legend()
axs[0].grid(True)

# CDF
sorted_delays, cdf_emp = empirical_cdf(delays)
axs[1].plot(sorted_delays, cdf_emp, 'b-', label='Empirical CDF')
axs[1].plot(x, cdf_theory, 'k--', label='Theoretical CDF')
axs[1].set_title('CDF of Total Delay (M/M/1)')
axs[1].set_xlabel('Delay')
axs[1].set_ylabel('CDF')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()