import numpy as np
import pandas as pd

def f(x, y):
    # optimization function
    return((1 - x)**2) + (100 * ((y - x**2)**2))

def c(x, y):#Inequality Contraints 
     c1 = max(0, ((x * x) + (x - y) + 1.5))
     c2 = max(0, (10 - (x * y)))
     return (c1,c2)

def constraint(x, y):#Fitness function with constraints penalty
    fit = f(x, y) + 15*sum(c(x, y))
    return(fit)

nBit = 10
XU,XL,YU,YL = 1, 0, 13, 0
popln = 20 ;Pc = 0.7 ; Pm = 1/popln
act_bit = int(nBit/2)

def CreatePopulation(popln, nBit):
    popl_binary = []
    for i in range(popln):
        binr1 = np.random.choice([0, 1], size=nBit)
        popl_binary.append(binr1)
    return popl_binary

Init_popl = pd.DataFrame(CreatePopulation(popln, nBit))

for k in range(50):
    X, Y = [], [] #value of X
    fit = [] #fitness
    for i in range(popln):   
        xx,yy = [], []
        for j in range(act_bit):
            xx.append(Init_popl.iloc[i][j] * (2 ** j))
            yy.append(Init_popl.iloc[i][act_bit+j] * (2 ** j))
        XX = XL +( (XU - XL) / ((2**act_bit)-1) ) * sum(xx)
        YY = YL +( (YU - YL) / ((2**act_bit)-1) ) * sum(yy)
        X.append(XX);Y.append(YY)
        fit.append(constraint(XX,YY))
    a1 = fit.index(min(fit))
    print("X = {}".format(X[a1]))
    print("Y = {}".format(Y[a1]))
    
    #Tournament selection
    tourn_popl = []
    for j in range(popln):
        ran1, ran2 = np.random.randint(0,popln-1,2) #Generating random int between 0 to 9 of size 2
        n = [fit[ran1],fit[ran2]]
        if n[0] < n[1]:
            tourn_popl.append(list(Init_popl.iloc[ran1]))
        else:
            tourn_popl.append(list(Init_popl.iloc[ran2]))
    tourn_popl = pd.DataFrame(tourn_popl)   

    #Crossover
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
    
    #Mutation
    for i in range(popln):
        mute = cross_popl.loc[i]
        for j in range(nBit):
            rnd_no = np.random.uniform(0,1)
            if rnd_no < Pm:
                if mute[j] == 0:
                    mute[j] = 1;break
                else:
                    mute[j] = 0;break
    Init_popl = cross_popl