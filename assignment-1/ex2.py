import numpy as np
import random
import matplotlib.pyplot as plt

class ExponentialRv:

    def __init__(self, param):
        
        if(param <= 0):
            raise ValueError("Parameter must be > 0")
        
        self.param = param

    def sample(self):
        return np.random.exponential(1 / self.param)

class UniformRv:

    def __init__(self, low, high):
        
        if(high <= low):
            raise ValueError("Parameter must be low < high")
        
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)


MEAN = 1
myexp = ExponentialRv(MEAN)

LOW, HIGH = 0, 5
myuni = UniformRv(LOW, HIGH)

N = 300_000
samples_exp = []
samples_uni = []

for i in range(N):
    samples_exp.append(myexp.sample())
    samples_uni.append(myuni.sample())

empirical_probability = sum(1 for x, y in zip(samples_exp, samples_uni) if x > y) / N

theoretical_probability = (1 / (HIGH - LOW)) * (1 - np.exp(-HIGH))

print(f"Empirical probability P(X > Y): {empirical_probability:.6f}")
print(f"Theoretical probability P(X > Y): {theoretical_probability:.6f}")


# Exponential Distribution
plt.figure(figsize=(10, 4))
plt.title('Distribution of the Exponential Samples (mean=1)')
plt.hist(samples_exp, bins=150, density=False, alpha=0.6, color='blue')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('./assignment-1/samples-exp.png')

# Uniform Distribution
plt.figure(figsize=(10, 4))
plt.title('Distribution of the Uniform Samples [0, 5]')
plt.hist(samples_uni, bins=150, density=False, alpha=0.6, color='red')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('./assignment-1/samples-uniform.png')