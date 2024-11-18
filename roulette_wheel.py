import numpy as np  
  
def RouletteWheelSelection(P):  
    r = np.random.rand()  
    C = np.cumsum(P)  
    i = np.where(C >= r)[0][0]  
    
    return i
