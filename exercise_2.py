import numpy as np
import matplotlib.pyplot as plt

#Class to model our exponential RV
class ExponentialRv:

    def __init__(self, param):
        
        if(param <= 0):
            raise ValueError("Parameter must be > 0")
        
        self.param = param

    def sample(self):
        return np.random.exponential(self.param)

#Class to model our uniform RV
class UniformRv:

    def __init__(self, low, high):
        
        if(high <= low):
            raise ValueError("Parameter must be low < high")
        
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)


MEAN = 1
myexp = ExponentialRv(1 / MEAN)

LOW, HIGH = 0, 5
myuni = UniformRv(LOW, HIGH)

#number of samples to draw
N = 5_000_000
samples_exp = []
samples_uni = []

for i in range(N):
    samples_exp.append(myexp.sample())
    samples_uni.append(myuni.sample())

#just count how many examples in the sample are actually greater
empirical_probability = sum(1 for x, y in zip(samples_exp, samples_uni) if x > y) / N

#see the formula on the report
theoretical_probability = (1 / (HIGH - LOW)) * (np.exp(LOW) - np.exp(-HIGH))

print(f"Empirical probability P(X > Y): {empirical_probability:.6f}")
print(f"Theoretical probability P(X > Y): {theoretical_probability:.6f}")