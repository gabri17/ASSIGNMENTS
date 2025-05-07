import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPARATION

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

#### REMOVING TREND FROM THE DATA
m = 5 #best degree 

p = np.polyfit(x, y, m)
yy = np.zeros(len(x))
for j in range(len(p)):
    k = len(p) - j - 1
    yy += p[j] * (x**k)

residuals = y - yy
y = residuals

########################################
#### IMPLEMENTING EXPECTATION MAXIMIZATION ALGORITHM

#Parameters initialization
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

print("Initial mu1, sigma1", mu1, sigma1)
print("Initial mu2, sigma2", mu2, sigma2)
print("Initial mu3, sigma3", mu3, sigma3)


#Hyperparameters intialization
num_epochs = 120
PRINT_EVERY = 15
assignments = []

priors = [1/3, 1/3, 1/3]

N = len(y)

#function to compute the likelihood that a datapoint comes from a certain gaussian i
def compute_likelihood(mu, sigma, datapoint, i):
    numerator = f(datapoint, mu, sigma) * priors[i]
    denominator = f(datapoint, mu1, sigma1) * priors[0] + f(datapoint, mu2, sigma2) * priors[1] + f(datapoint, mu3, sigma3) * priors[2]
    return numerator / denominator


for epoch in range(num_epochs):

    if epoch % PRINT_EVERY == 0:
        print("\tEpoch: ", epoch)
        print("\tmu1, sigma1", mu1, sigma1)
        print("\tmu2, sigma2", mu2, sigma2)
        print("\tmu3, sigma3", mu3, sigma3)
        print("\tpriors", priors)
        print("\n")

    assignments = []
    
    #E-step
    c1, c2, c3 = 0, 0, 0
    for i in range(N):
        first = compute_likelihood(mu1, sigma1, y[i], 0) #likelihood of y[i] comes from gaussian 1
        second = compute_likelihood(mu2, sigma2, y[i], 1)
        third = compute_likelihood(mu3, sigma3, y[i], 2)
        
        c1 += first
        c2 += second
        c3 += third

        assignments.append((first, second, third))
    
    #M-step
    mu1, mu2, mu3 = 0, 0, 0
    sigma1, sigma2, sigma3 = 0, 0, 0
    
    for i in range(N):
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


###PLOTTING THE RESULTS
fig = plt.figure()

BINS=200
plt.hist(residuals, bins=BINS, color='red', alpha=0.7, density=True)
plt.xlabel('Metric values')
plt.ylabel('Density')
plt.title('Histogram of data WITHOUT trend')

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
print("Final mu1, sigma1", mu1, sigma1)
print("Final mu2, sigma2", mu2, sigma2)
print("Final mu3, sigma3", mu3, sigma3)
print("Actual mu1, sigma1", real_mu1, real_sigma1)
print("Actual mu2, sigma2", real_mu2, real_sigma2)
print("Actual mu3, sigma3", real_mu3, real_sigma3)

plt.tight_layout()
plt.show()

#Few discrepancies betwen the estimated and real distributions used in generating the data
#Increasing number of epochs we could achieve a better fit
#Additionally, it is important to note that it's very difficult to assign every point correctly to the right gaussian
#Since maybe in the generation process it belongs to the tail of one distribution, while maybe here is estiamted to belong to another one
#Anyway, reasoning in terms of "classification", this can be considered a good and desirable result, since maybe the point was an outlier in the generation process