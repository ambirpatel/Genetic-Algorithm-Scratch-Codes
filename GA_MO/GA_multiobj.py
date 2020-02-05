import numpy as np
import pandas as pd

def F1(x):
    return (x*x*x)
    
def F2(x):
    return (x-2)**2

nBit = 8
XU, XL = 10e2, -10e2
popln = 10 ;Pc = 0.7 ; Pm = 1/popln

def Population_generation(popln, nBit):
    popl_binary = []
    for i in range(popln):
        binr1 = np.random.choice([0, 1], size=nBit)
        popl_binary.append(binr1)
    return popl_binary

Init_popl = pd.DataFrame(Population_generation(popln, nBit))

def mapping(Init_popl):
    X, f1, f2 = [], [], []
    for i in range(popln):   
        xx = []
        for j in range(nBit):
            xx.append(Init_popl.iloc[i][j] * (2 ** j))
        XX = XL +( (XU - XL) / ((2**nBit)-1) ) * sum(xx)
        X.append(XX)
        f1.append(F1(XX)); f2.append(F2(XX))
    return zip(X, f1, f2)

Fitness = pd.DataFrame(mapping(Init_popl));Fitness.columns = ['x', 'f1', 'f2']

#Tournament selection
tourn_popl = []
for j in range(popln):
    ran1, ran2 = np.random.randint(0,popln-1,2) #Generating random int between 0 to 9 of size 2
    n = [Fitness.iloc[ran1, 1], Fitness.iloc[ran2, 1]]
    if n[0] < n[1]:
        tourn_popl.append(list(Init_popl.iloc[ran1]))
    else:
        tourn_popl.append(list(Init_popl.iloc[ran2]))
tourn_popl = pd.DataFrame(tourn_popl)

#CrossOver
cross_popl = []
for i in range(int(popln/2)): 
    ran1, ran2 = np.random.randint(0,popln-1,2)
    crX1, crX2 = list(tourn_popl.iloc[ran1]), list(tourn_popl.iloc[ran2])
    
    for j in range(nBit):
        rn = np.random.uniform(0,1)
        if rn < Pc: 
            break
    crossX1, crossX2 = crX1[0:j] + crX2[j:], crX2[0:j] + crX1[j:]
    cross_popl.append(crossX1), cross_popl.append(crossX2)
cross_popl = pd.DataFrame(cross_popl)

NSGA_pop = Init_popl.append(cross_popl)
nf = pd.DataFrame(mapping(cross_popl));nf.columns = ['x', 'f1', 'f2']
NSGA_fit = Fitness.append(nf)
