import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPARATION

data_file = 'data_ex1_wt.csv' # absolute path respect to when the script is run
df = pd.read_csv(data_file, header=None, names=['time', 'metric'])

x = df['time'].values
y = df['metric'].values

# pdf of a gaussian
def f(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
#

#### REMOVING TREND FROM THE DATA
m = 5 #best degree

p = np.polyfit(x, y, m)
yy = np.zeros(len(x))
for j in range(len(p)):
    k = len(p) - j - 1
    yy += p[j] * (x**k)

residuals = y - yy
y = residuals

###########################################
# EM algorithm to fit a mixture of gaussians to the data

def em(data, gaussians, num_epochs=90, print_every=15):

    #Auxiliary functions
    ###################################

    #random value in an internval [low, high] - we initialize in this way to speed up the convergence
    def get_random(low=-np.std(data), high=np.std(data)):
        return np.random.rand() * (high-low) + low

    #pdf of a gaussian
    def f(x, mu, sigma):
        return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    
    #internal function to print parameters
    def print_parameters():
        for k in range(gaussians):
            print("\tmu", k, "sigma", k, parameters[k][0], parameters[k][1])
        print("\tPriors: ", priors)
        print("\n")

    #internal function to compute the likelihood of datapoint coming from gaussian i
    def compute_likelihood(mu, sigma, datapoint, i):
        numerator = f(datapoint, mu, sigma) * priors[i]
        denominator = 0
        for k in range(gaussians):
            denominator += f(datapoint, parameters[k][0], parameters[k][1]) * priors[k]
        return numerator / denominator

    #internal function to reset the parameters
    def reset():
        for k in range(gaussians):
            parameters[k] = [0, 0]

    ###################################

    #Parameters initialization
    start1 = np.mean(data)
    start2 = np.std(data)

    parameters = []
    priors = []


    #initialize the parameters of the gaussians
    for _ in range(gaussians):
        parameters.append([start1 + get_random(), start2])
    
    #initialize the priors of the gaussians
    for _ in range(gaussians):
        priors.append(1/gaussians)

    N = len(data)

    for epoch in range(num_epochs):

        if print_every != 0 and epoch % print_every == 0:
            print("Epoch: ", epoch)
            print_parameters()

        assignments = []
        
        #E-step
        for i in range(N):
            assignments.append([])
            #for each gaussian, compute the likelihood of the datapoint coming from that gaussian
            for k in range(gaussians):
                assignments[i].append(compute_likelihood(parameters[k][0], parameters[k][1], data[i], k))

    
        #M-step
        reset()
        counts = [0] * gaussians
    
        #update the parameters of the gaussians with various iterations (to compute sum of the likelihoods abd mean)
        for i in range(N):
            for k in range(gaussians):
                counts[k] += assignments[i][k]
                parameters[k][0] += assignments[i][k] * data[i]

        for k in range(gaussians):
            parameters[k][0] /= counts[k]
        
        #std dev computation
        for i in range(N):
            for k in range(gaussians):
                parameters[k][1] += assignments[i][k] * ((data[i] - parameters[k][0])**2)

        for k in range(gaussians):
            parameters[k][1] /= counts[k]
            parameters[k][1] = np.sqrt(parameters[k][1])

        #update the priors of the gaussians
        for k in range(gaussians):
            priors[k] = counts[k] / N
    
    return parameters, assignments, priors

################################################
points = np.linspace(-11, 11, 1000)

plot = True #change if you want to plot the results or not - it enables a "visual comparison" of the gaussians
dataset_likelihood = []

MINIMUM_GAUSSIANS = 2
MAXIMUM_GAUSSIANS = 5

def compute_log_likelihood(data, priors, params):
    N = len(data)
    k = len(priors)
    log_likelihood = 0.0
    for i in range(N):
        total = 0.0
        for j in range(k):
            total += priors[j] * f(data[i], params[j][0], params[j][1])
        log_likelihood += np.log(total)
    return log_likelihood

for gaussian in range(MINIMUM_GAUSSIANS, MAXIMUM_GAUSSIANS + 1):
    param, likelihoods, priors = em(residuals, gaussian, num_epochs=150, print_every=0)
    
    #likelihood is p(x|mu,sigma) * p(mu,sigma) = p(mu,sigma|x)

    L = compute_log_likelihood(residuals, priors, param)
    aic = -2 * L + 2 * (gaussian * 2)
    dataset_likelihood.append(aic)

    print("Gaussians: ", gaussian)
    print("\tDataset AIC: ", dataset_likelihood[-1])
    for k in range(gaussian):
        print(f"\t[{k} Gaussian] ({param[k][0]}, {param[k][1]})")
    
    if(plot):
        #For visual comparison
        fig = plt.figure()

        BINS=200
        plt.hist(residuals, bins=BINS, color='red', alpha=0.7, density=True)

        colors = [
            'blue', 'red', 'green', 'orange', 'purple', 
            'cyan', 'magenta', 'brown', 'olive', 'pink', 
            'gray', 'teal', 'navy', 'maroon', 'lime'
        ]

        for k in range(gaussian):
            mu, sigma = param[k][0], param[k][1]
            plt.plot(points, list(map(lambda x: f(x, mu, sigma), points)), color=colors[k % len(colors)], linestyle='dashed')
        
        plt.title('EM with ' + str(gaussian) + ' Gaussians')
        plt.tight_layout()
        plt.show()

##### PLOTTING THE RESULTS
#top model is the one with the lowest AIC
fig = plt.figure()
plt.plot(range(1, len(dataset_likelihood)+1), dataset_likelihood, 'o-', color='red')
plt.title('AIC vs Gaussian')
plt.xlabel('#Gaussians')
plt.xticks(range(1, len(dataset_likelihood)+1))
plt.ylabel('Dataset AIC')
plt.show()
################################################
"""
Target:
real_mu1, real_sigma1^2 = -5, 3
real_mu2, real_sigma2^2 = 0, 6
real_mu3, real_sigma3^2 = 4, 1
"""