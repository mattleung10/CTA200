#For Python 3
#System of Ordinary Differential Equations: SIR epidemiology model

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

"""
The system of ODEs that we will solve is (SIR model):
    S'(t) = -b*S(t)*I(t)/N
    I'(t) = b*S(t)*I(t)/N - g*I(t)
    R'(t) = g*I(t)
    Constants: b, N, g
Initial Conditions:
    S(0) = 999
    I(0) = 1
    R(0) = 0
Interval:
    0 <= t <= 200
"""

def SIR(y,t,b,N,g):
    S, I, R = y
    #S = number of susceptible
    #I = number of infections
    #R = number of recovered or deceased
    #N is the total population, b is the average number of contacts per person per time
    #g is transition (recovery) rate
    #https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
    
    dydt = [-1*b*S*I/N,
            b*S*I/N - g*I,
            g*I]
    return dydt

def SIRD(y,t,b,N,g,m):
    S, I, R, D = y
    #S = number of susceptible
    #I = number of infections
    #R = number of recovered
    #D = number of deaths
    #N is the total population, b is the average number of contacts per person per time
    #g is recovery rate, m is the mortality rate
    #https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model
    
    dydt = [-1*b*S*I/N,
            b*S*I/N - g*I - m*I,
            g*I,
            m*I]
    return dydt

def solve_SIRmodel(b,N,g,ti,tf,y0_SIR,solve_SIR=True,m=0,y0_SIRD=[]):
    """
    INPUT:
    b, N, g, ti, tf, y0_SIRD        #see SIR function
    OPTIONAL INPUT:
    ::boolean:: solve_SIR           #whether or not to solve SIR, default True; if False, solve for SIRD system
    m, y0_SIRD                      #see SIRD function
    """
    t = np.linspace(ti, tf, 5000)
    if solve_SIR == True:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
        #SIR model:
        sol_SIR = scipy.integrate.odeint(SIR, y0_SIR, t, args=(b, N, g))
        plt.title('SIR Model')
        plt.plot(t, sol_SIR[:,0], 'y', label='S(t)')
        plt.plot(t, sol_SIR[:,1], 'r', label='I(t)')
        plt.plot(t, sol_SIR[:,2], 'b', label='R(t)')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()
    else:
        #SIRD model (includes death rate):
        sol_SIRD = sol_SIR = scipy.integrate.odeint(SIRD, y0_SIRD, t, args=(b, N, g, m))
        plt.title('SIRD Model')
        plt.plot(t, sol_SIRD[:,0], 'y', label='S(t)')
        plt.plot(t, sol_SIRD[:,1], 'r', label='I(t)')
        plt.plot(t, sol_SIRD[:,2], 'b', label='R(t)')
        plt.plot(t, sol_SIRD[:,3], 'black', label='D(t)')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.show()

def main():
    b = 0.2
    N = 1000
    g = 0.1 
    
    y0_SIR = [999,1,0] #Initial Conditions for SIR model
    
    m = 0.02
    y0_SIRD = [999,1,0,0] #Initial Conditions for SIRD model
    
    #set of b and g values to plot
    b_set = [0.2,0.5,0.12,0.4]
    g_set = [0.1,0.1,0.07,0.3]
    try:
        assert(len(b_set) == len(g_set)) #check if both lists are same length
    except AssertionError:
        print("ERROR: Lists not the same length")
        return False
    
    #Interval
    ti = 0
    tf = 200
    
    for i in range(0,len(b_set),1):
        solve_SIRmodel(b_set[i],N,g_set[i],ti,tf,y0_SIR)

    solve_SIRmodel(b,N,g,ti,tf,y0_SIR,False,m,y0_SIRD)

if __name__ == "__main__":
    main()
