import numpy as np  
from portfolio_moc import PortMOC

def PortCost(x, model):
    DesiredRet = model['DesiredRet']  
    _, out = PortMOC(x, model)  
    rsk = out['rsk']  
    ret = out['ret']  
    viol = max(0, 1 - ret / DesiredRet)  
    beta = 1000  
    z = rsk * (1 + beta * viol)  
    out['viol'] = viol  
    out['IsFeasible'] = (viol == 0) 
    
    return z, out