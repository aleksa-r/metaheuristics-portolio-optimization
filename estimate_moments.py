import numpy as np  
  
    
def EstimateReturnMoments(R, Semi=False):  
    MU = np.mean(R, axis=0).reshape(-1, 1)  
    n = R.shape[1]  
  
    if not Semi:
        SIGMA = np.cov(R, rowvar=False)  
    else:
        sigma = np.zeros((n, 1))  
        
        for i in range(n):
            dev = R[:, i] - MU[i]
            sigma[i] = np.sqrt(np.mean(dev[dev < 0]**2))
        rho = np.corrcoef(R, rowvar=False)  
        SIGMA = rho  
        for i in range(n):
            SIGMA[i, :] *= sigma[i]  
            SIGMA[:, i] *= sigma[i]  
  
    return MU, SIGMA
