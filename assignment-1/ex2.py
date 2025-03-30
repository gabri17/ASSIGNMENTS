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

#Playing with the plots
plt.title('Distribution of the samples of the exponential')
plt.hist(samples_exp, bins=150, density=False, alpha=0.6, color='blue') #bins to group data, alpha for aesthetic, density = True for normalization and does not show actual values
plt.show()
#plt.savefig('./assignment-1/samples-exp.png')

plt.title('Distribution of the samples of the uniform')
plt.hist(samples_uni, bins=150, density=False, alpha=0.6, color='red')
plt.show()
#plt.savefig('./assignment-1/samples-uniform.png')
