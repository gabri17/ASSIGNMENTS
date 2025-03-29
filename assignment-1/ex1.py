import numpy as np
import random

import matplotlib.pyplot as plt

"""
mean_1, var_1 = -2, 2
mean_2, var_2 = 4, 1
mean_3, var_3 = 10, 3
mean_4, var_4 = 15, 2

p_1 = 0.15
f_1 = 0.15
p_2 = 0.25
f_2 = 0.40
p_3 = 0.35
f_3 = 0.75
p_4 = 0.25
f_4 = 1
"""

class ConditionalGaussian:

    def __init__(self, means, variances, probabilities):
        if(sum(probabilities) != 1):
            print("Probabilities do not respect probabilities axioms")

        if(len(probabilities) != len(means) & len(means) != len(variances)):
            print("Not ok data")
        
        self.means = means
        self.variances = variances
        self.probabilities = probabilities
        f = []
        for i in range(len(self.probabilities)):
            if i != 0:
                f.append(f[i-1] + self.probabilities[i])
            else:
                f.append(self.probabilities[i])
        
        self.f = f

    
    def get_conditioanl_mean(self):
        return np.dot(self.means, self.probabilities)
    
    def get_conditional_variance(self):
        expected_variance = np.dot(self.variances, self.probabilities)
        computed_mean = self.get_conditioanl_mean()
        variance_of_expectation = np.dot(list(map(lambda x: (x-computed_mean)**2, self.means)), self.probabilities)
        return expected_variance + variance_of_expectation
    
    def sample(self):
        U = random.random()

        for i in range(len(self.f)):
            if i == 0:
                if(U <= self.f[i]):
                    return float(np.random.normal(self.means[i], np.sqrt(self.variances[i]), 1)[0])
            else:
                if((U > self.f[i-1]) & (U <= self.f[i])):
                    return float(np.random.normal(self.means[i], np.sqrt(self.variances[i]), 1)[0])
                else:
                    continue

myvar = ConditionalGaussian([-2, 4, 10, 15], [2, 1, 3, 2], [0.15, 0.25, 0.35, 0.25])
print("Theoretical mean: " + str(myvar.get_conditioanl_mean()))
print("Theoretical variance: " + str(myvar.get_conditional_variance()))

N = 1_000_000

samples = []
means, variances = [], []
examples = []

offset = 10_000

for i in range(N):
    samples.append(myvar.sample())
    
    if(i < offset):
        examples.append(i)
        means.append(float(np.mean(samples)))
        variances.append(np.var(samples))

yticks1 = 1
yticks2 = 1

plt.title('Means against number of examples')
plt.xlim(0, offset)
plt.ylim(0, myvar.get_conditioanl_mean()*2)
plt.yticks(np.arange(0, myvar.get_conditioanl_mean()*2, yticks1))
plt.plot(examples, means)
plt.savefig('./assignment-1/mean.png')


plt.title('Variances against number of examples')
plt.xlim(0, offset)
plt.ylim(0, myvar.get_conditional_variance()*(3/2))  
#plt.yticks(np.arange(0, myvar.get_conditional_variance()*2, yticks2))
plt.plot(examples, variances)
plt.savefig('./assignment-1/variance.png')


#print(samples)
empirical_mean_dataset, empirical_var_dataset = np.mean(samples), np.var(samples)
print("Dataset mean: " + str(empirical_mean_dataset), "\nDataset variance: " + str(empirical_var_dataset))

"""
computed_mean = mean_1 * p_1 + mean_2 * p_2 + mean_3 * p_3 + mean_4 * p_4
print("Computed mean: " + str(computed_mean))

computed_variance = (p_1*var_1 + p_2*var_2 + p_3*var_3 + p_4*var_4) + (p_1*((mean_1-computed_mean)**2) + p_2*((mean_2-computed_mean)**2) + p_3*((mean_3-computed_mean)**2) + p_4*((mean_4-computed_mean)**2))
print("Computed variance: " + str(computed_variance))
"""


#Var(X) = E[Var(X|Y)] + Var(E[X|Y])
#X is my conditional distribution
#Y is the probability of each gaussian
#X | Y is one of the gaussians

#E[X] = E[X|Y1]P(Y1) + E[X|Y2]P(Y2) + E[X|Y3]P(Y3) + E[X|Y4]P(Y4)
#E[X|Y] is given
#E[Var(X|Y)] = p_1*var_1 + p_2*var_2 + p_3*var_3 + p_4*var_4
#Var(E[X|Y]) = p_1*((mean_1-computed_mean)**2) + p_2*((mean_2-computed_mean)**2) + p_3*((mean_3-computed_mean)**2) + p_4*((mean_4-computed_mean)**4)