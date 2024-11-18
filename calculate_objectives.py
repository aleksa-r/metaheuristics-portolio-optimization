import numpy as np  
from estimate_moments import EstimateReturnMoments
    
def CalculatePortfolioObjectives(w, R, method='cvar', alpha=0.95):
    w = w.reshape(-1, 1)  
  
    if method.lower() == 'mad':
        port = PortfolioMAD()  
        port.setScenarios(R)  
    elif method.lower() == 'cvar':
        port = PortfolioCVaR()  
        port.setScenarios(R)  
        port.setProbabilityLevel(alpha)  
    else:  
        Semi = method.lower() == 'msv'  
        MU, SIGMA = EstimateReturnMoments(R, Semi)  
        port = Portfolio()  
        port.setAssetMoments(MU, SIGMA)  
  
    rsk = port.estimatePortRisk(w)  
    ret = port.estimatePortReturn(w)
    
    return rsk, ret


class PortfolioCVaR:
    
    def __init__(self):
        self.scenarios = None
        self.probability_level = None  
  
    def setScenarios(self, scenarios):
        self.scenarios = scenarios  
  
    def setProbabilityLevel(self, probability_level):
        self.probability_level = probability_level  
  
    def estimatePortRisk(self, w):
        # Calculate the portfolio returns  
        port_returns = np.dot(self.scenarios, w)  
  
        # Calculate the VaR  
        var = np.percentile(port_returns, (1 - self.probability_level) * 100)  
  
        # Calculate the CVaR  
        cvar = np.mean(port_returns[port_returns <= var])  
  
        return cvar  
  
    def estimatePortReturn(self, w):
        # Calculate the portfolio returns  
        port_returns = np.dot(self.scenarios, w)  
  
        # Calculate the mean return  
        mean_return = np.mean(port_returns)  
  
        return mean_return
 
    
class PortfolioMAD:
    def __init__(self):
        self.scenarios = None  
  
    def setScenarios(self, scenarios):
        self.scenarios = scenarios  
  
    def estimatePortRisk(self, w):
        # Calculate the portfolio returns  
        port_returns = np.dot(self.scenarios, w)  
  
        # Calculate the Mean Absolute Deviation (MAD)  
        mad = np.mean(np.abs(port_returns - np.mean(port_returns)))  
  
        return mad  
  
    def estimatePortReturn(self, w):
        # Calculate the portfolio returns
        port_returns = np.dot(self.scenarios, w)  
  
        # Calculate the mean return  
        mean_return = np.mean(port_returns)  
  
        return mean_return

  
class Portfolio:  
    def __init__(self):
        self.asset_moments = None  
  
    def setAssetMoments(self, mu, sigma):
        self.asset_moments = (mu, sigma)  
  
    def estimatePortRisk(self, w):
        # Calculate the portfolio variance  
        port_var = np.dot(w.T, np.dot(self.asset_moments[1], w))  
  
        # Calculate the portfolio standard deviation  
        port_std = np.sqrt(port_var)  
  
        return port_std  
  
    def estimatePortReturn(self, w):  
        # Calculate the portfolio return  
        port_return = np.dot(w.T, self.asset_moments[0])  
  
        return port_return
