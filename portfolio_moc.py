import numpy as np  
from calculate_objectives import CalculatePortfolioObjectives

def PortMOC(x, model):
    R = model['R']  
    method = model.get('method', '')  
    alpha = model.get('alpha', '')  
    w = x / np.sum(x)  
    rsk, ret = CalculatePortfolioObjectives(w, R, method, alpha)  
    z = np.array([rsk, -ret])  
    out = {'w': w, 'rsk': rsk, 'ret': ret}  
    
    return z, out
