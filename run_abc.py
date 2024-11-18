import numpy as np  
from portfolio_cost import PortCost
from roulette_wheel import RouletteWheelSelection
  
def RunABC(model):
    # Problem  
    CostFunction = lambda x: PortCost(x, model)  
    nVar = model['R'].shape[1]  
    VarSize = (1, nVar)  
    VarMin = 0.1  
    VarMax = 0.3  
  
    # ABC Settings  
    MaxIt = 200
    nPop = 50
    nOnlooker = nPop  
    L = round(0.6 * nVar * nPop)  
    a = 1  
  
    # Initialization  
    empty_bee = {'Position': None, 'Cost': None, 'Out': None}  
    pop = [empty_bee.copy() for _ in range(nPop)]  
    BestSol = {'Cost': float('inf')}  
  
    for i in range(nPop):
        pop[i]['Position'] = np.random.uniform(VarMin, VarMax, VarSize)  
        pop[i]['Cost'], pop[i]['Out'] = CostFunction(pop[i]['Position'])  
        if pop[i]['Cost'] <= BestSol['Cost']:  
            BestSol = pop[i]  
  
    C = np.zeros(nPop)  
    BestCost = np.zeros(MaxIt)  
  
    # ABC Main Loop  
    for it in range(MaxIt):  
       # Recruited Bees  
       for i in range(nPop):
            K = list(range(nPop))  
            K.remove(i)  
            k = np.random.choice(K)  
            phi = a * np.random.uniform(-1, 1, VarSize)  
            newbee = {'Position': pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])}  
            newbee['Position'] = np.clip(newbee['Position'], VarMin, VarMax)  
            newbee['Cost'], newbee['Out'] = CostFunction(newbee['Position'])  
            
            if newbee['Cost'] <= pop[i]['Cost']:
                pop[i] = newbee  
            else:
                C[i] += 1  
  
       # Calculate Fitness Values and Selection Probabilities  
       F = np.exp(-np.array([bee['Cost'] for bee in pop]) / np.mean([bee['Cost'] for bee in pop]))
       P = F / np.sum(F)
  
       # Onlooker Bees  
       for m in range(nOnlooker):
            i = RouletteWheelSelection(P)  
            K = list(range(nPop))  
            K.remove(i)  
            k = np.random.choice(K)  
            phi = a * np.random.uniform(-1, 1, VarSize)  
            newbee = {'Position': pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])}  
            newbee['Position'] = np.clip(newbee['Position'], VarMin, VarMax)  
            newbee['Cost'], newbee['Out'] = CostFunction(newbee['Position'])  
            if newbee['Cost'] <= pop[i]['Cost']:
                pop[i] = newbee  
            else:
                C[i] += 1  
  
       # Scout Bees  
       for i in range(nPop):
            if C[i] >= L:
                pop[i]['Position'] = np.random.uniform(VarMin, VarMax, VarSize)
                pop[i]['Cost'], pop[i]['Out'] = CostFunction(pop[i]['Position'])  
                C[i] = 0  
  
       # Update Best Solution Ever Found  
       for i in range(nPop):
            if pop[i]['Cost'] <= BestSol['Cost']:
                BestSol = pop[i]  
  
       # Store Best Cost Ever Found  
       BestCost[it] = BestSol['Cost']  
  
       # Display Iteration Information  
       print(f'Iteration {it+1}: Best Cost = {BestCost[it]}')  
  
   # Export Results  
    out = {'BestSol': BestSol, 'BestCost': BestCost}
    
    return out
    