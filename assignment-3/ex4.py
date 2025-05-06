import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

m = 5 #best degree 

p = np.polyfit(x, y, m)
yy = np.zeros(len(x))
for j in range(len(p)):
    k = len(p) - j - 1
    print(k, p[j])
    yy += p[j] * (x**k)

residuals = y - yy

y = residuals

###########################################

LOW, HIGH = -3, 3

def get_random(low=LOW, high=HIGH):
    return np.random.rand() * (high-low) + low

def f(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

start1 = np.mean(y)
start2 = np.std(y)

mu1, sigma1 =  start1 + get_random(),  start2
mu2, sigma2 =  start1 + get_random(),  start2
mu3, sigma3 =  start1 + get_random(),  start2

print("FIRST mu1, sigma1", mu1, sigma1)
print("FIRST mu2, sigma2", mu2, sigma2)
print("FIRST mu3, sigma3", mu3, sigma3)


#########################
num_epochs = 90
assignments = []

priors = [1/3, 1/3, 1/3]

N = len(y)

def compute_likelihood(mu, sigma, el, i):
    numerator = f(el, mu, sigma) * priors[i]
    denominator = f(el, mu1, sigma1) * priors[0] + f(el, mu2, sigma2) * priors[1] + f(el, mu3, sigma3) * priors[2]
    return numerator / denominator

PRINT_EVERY = 15

for epoch in range(num_epochs):

    if epoch % PRINT_EVERY == 0:
        print("Epoch: ", epoch)
        print("mu1, sigma1", mu1, sigma1)
        print("mu2, sigma2", mu2, sigma2)
        print("mu3, sigma3", mu3, sigma3)
        print("priors", priors)

    assignments = []
    
    #E-step
    for i in range(N):
        first = compute_likelihood(mu1, sigma1, y[i], 0)
        second = compute_likelihood(mu2, sigma2, y[i], 1)
        third = compute_likelihood(mu3, sigma3, y[i], 2)
        assignments.append((first, second, third))
    
    #M-step
    mu1, mu2, mu3 = 0, 0, 0
    c1, c2, c3 = 0, 0, 0
    sigma1, sigma2, sigma3 = 0, 0, 0
    
    for i in range(N):
        c1 += assignments[i][0]
        c2 += assignments[i][1]
        c3 += assignments[i][2]

        mu1 += assignments[i][0] * y[i]
        mu2 += assignments[i][1] * y[i]
        mu3 += assignments[i][2] * y[i]

    mu1 /= c1
    mu2 /= c2  
    mu3 /= c3

    for i in range(N):
        sigma1 += assignments[i][0] * ((y[i] - mu1)**2)
        sigma2 += assignments[i][1] * ((y[i] - mu2)**2)
        sigma3 += assignments[i][2] * ((y[i] - mu3)**2)
    
    sigma1 /= c1
    sigma2 /= c2
    sigma3 /= c3

    sigma1 = np.sqrt(sigma1)
    sigma2 = np.sqrt(sigma2)
    sigma3 = np.sqrt(sigma3)

    #priors update
    priors[0] = c1 / N
    priors[1] = c2 / N
    priors[2] = c3 / N
    #print("New priors: ", priors)


################################################
fig = plt.figure()

BINS=200
plt.hist(residuals, bins=BINS, color='red', alpha=0.7, density=True)

real_mu1, real_sigma1 = -5, np.sqrt(3)
real_mu2, real_sigma2 = 0, np.sqrt(6)
real_mu3, real_sigma3 = 4, np.sqrt(1)


points = np.linspace(-11, 11, 1000)
plt.plot(points, list(map(lambda x: f(x, mu1, sigma1), points)), color='green', linestyle='dashed')
plt.plot(points, list(map(lambda x: f(x, mu2, sigma2), points)), color='orange', linestyle='dashed')
plt.plot(points, list(map(lambda x: f(x, mu3, sigma3), points)), color='purple', linestyle='dashed')
plt.plot(points, list(map(lambda x: f(x, real_mu1, real_sigma1), points)), color='black')
plt.plot(points, list(map(lambda x: f(x, real_mu2, real_sigma2), points)), color='brown')
plt.plot(points, list(map(lambda x: f(x, real_mu3, real_sigma3), points)), color='gray')
print("LAST mu1, sigma1", mu1, sigma1)
print("LAST mu2, sigma2", mu2, sigma2)
print("LAST mu3, sigma3", mu3, sigma3)
print("REAL mu1, sigma1", real_mu1, real_sigma1)
print("REAL mu2, sigma2", real_mu2, real_sigma2)
print("REAL mu3, sigma3", real_mu3, real_sigma3)

plt.title('Histogram of new distribution')

plt.tight_layout()
plt.show()
################################################
"""
Target:
real_mu1, real_sigma1 = -5, 3
real_mu2, real_sigma2 = 0, 6
real_mu3, real_sigma3 = 4, 1
"""