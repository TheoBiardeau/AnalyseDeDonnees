import numpy as np


Data = np.array([[89,114,111,44.,10.4,57.,75.,106.,10.,],
                 [50.,58.5,75.4,5.3,0.,20.9,23.9,55.1,0.9]], dtype=float)
print(Data.shape)
def my_acp (X):
    #Algorytme to realyze acp. Warning, study varaible are on x axis (n)


    #setup all variable
    n , p = np.shape(X)
    average_value = np.zeros(n)
    C = np.zeros((n,p))
    inertia = np.zeros(n)
    tc = np.zeros((n, p))
    tc[:,:] = X[:,:]

    #compute average value of each ligne
    for i in range (n):
        average_value[i] = np.average(X[i,:])

    #compute matrix centré
    for i in range(n):
        tc[i,:] = tc[i,:] - average_value[i]
    #compute variance covariance matrix
    Mcov = np.cov(tc)
    #compute eigen value and eigen vector
    u, lamb = np.linalg.eigh(Mcov.T)
    #compute intertia
    for i in range (n):
        inertia[i] = u[i]/(np.sum(u))

    #compute projection on factoriel axe
    for i in range (n):
        for j in range(p):
            C[i,j] = np.dot(lamb[i], tc[:,j])

    return (lamb,C,u)

my_acp(Data)